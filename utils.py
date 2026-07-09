# Operacje na danych
import numpy as np
import pandas as pd
import time

# TensorFlow
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import logging
tf.get_logger().setLevel(logging.ERROR)

# Rysowanie wykresów
import matplotlib.pyplot as plt

# Zwiększenie rozmiaru czcionek na wykresach
params = {
    'legend.fontsize': 16,
    'figure.figsize': (10, 7),
    'axes.labelsize': 'xx-large',
    'axes.titlesize': 'xx-large',
    'xtick.labelsize': 'xx-large',
    'ytick.labelsize': 'xx-large'
}
plt.rcParams.update(params)


class PositionalEncoding2D(tf.keras.layers.Layer): 
    def __init__(self, max_height=32, max_width=32, **kwargs):
        super().__init__(**kwargs)
        self.max_height = max_height
        self.max_width = max_width

    def build(self, input_shape):
        _, H, W, C = input_shape

        # oddzielnie dla wysokości i szerokości
        positions_row = np.arange(self.max_height, dtype=np.float32).reshape(-1, 1)  # (H, 1)
        positions_col = np.arange(self.max_width, dtype=np.float32).reshape(-1, 1)   # (W, 1)
        
        frequencies_row =  1.0/np.power(10000, np.arange(0, C ,2, dtype=np.float32)/ C)
        frequencies_row = np.expand_dims(frequencies_row, 0)
        frequencies_col =  1.0/np.power(10000, np.arange(0, C ,2, dtype=np.float32)/ C)
        frequencies_col = np.expand_dims(frequencies_col, 0)
        
        angles_row = positions_row / frequencies_row
        angles_col = positions_col / frequencies_col + np.pi/4  # asymetryczne kodowanie pozycyjne
    
        self.pe_row = np.zeros((self.max_height, C), dtype=np.float32)
        self.pe_row[:, 0::2] = np.sin(angles_row)
        self.pe_row[:, 1::2] = np.cos(angles_row)

        self.pe_col = np.zeros((self.max_width, C), dtype=np.float32)
        self.pe_col[:, 0::2] = np.sin(angles_col)
        self.pe_col[:, 1::2] = np.cos(angles_col)

        self.pe_row = tf.constant(self.pe_row.reshape(1, self.max_height, 1, C), dtype=tf.float32)
        self.pe_col = tf.constant(self.pe_col.reshape(1, 1, self.max_width, C), dtype=tf.float32)

    def call(self, X):
        H = tf.shape(X)[1]
        W = tf.shape(X)[2]
        X = X + self.pe_row[:, :H, :, :] + self.pe_col[:, :, :W, :]
        return X


class ConstraintsLayer(tf.keras.layers.Layer):
    """Warstwa narzucająca ograniczenia liniowe między zmiennymi docelowymi."""
    def __init__(self, eps=0.001, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, inputs):
        out_green, out_dead, out_clover, out_gdm, out_total = inputs
        green = tf.exp(out_green) - self.eps
        dead = tf.exp(out_dead) - self.eps
        clover = tf.exp(out_clover) - self.eps
        gdm = tf.exp(out_gdm) - self.eps
        total = tf.exp(out_total) - self.eps

        constraint1 = tf.reduce_mean(tf.square(gdm - (green + clover)))
        constraint2 = tf.reduce_mean(tf.square(total - (green + dead + clover)))
        constraint_loss = 0.4 * constraint1 + 0.4 * constraint2

        self.add_loss(constraint_loss)
        # Zwraca konkatenację, aby warstwa była częścią grafu
        return tf.concat([out_green, out_dead, out_clover, out_gdm, out_total], axis=-1)


def show_full_grid(dataset, title, cols=5, eps=0.001):
    for patches_flat, _ in dataset.take(1):
        patches = patches_flat[0].numpy()
        num_patches, H, W, C = patches.shape
        
        rows = (num_patches + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
        # Dostosowanie axes do jednowymiarowej iteracji
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx in range(num_patches):
            i = idx // cols
            j = idx % cols
            axes[i, j].imshow(tf.image.hsv_to_rgb(patches[idx]))
            axes[i, j].axis('off')
            axes[i, j].set_title(f"{i},{j}", fontsize=8)
        
        for idx in range(num_patches, rows * cols):
            i = idx // cols
            j = idx % cols
            axes[i, j].axis('off')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()
        return


def find_best_stride(H, W, patch_size, stride_x, stride_y, min_stride=1):
    diff_h = H - patch_size
    diff_w = W - patch_size
    
    def closest_divisor(diff, target):
        best = min_stride
        best_dist = abs(best - target)
        for s in range(min_stride, diff + 1):
            if diff % s == 0:
                dist = abs(s - target)
                if dist < best_dist:
                    best_dist = dist
                    best = s
        return best
    
    stride_y = closest_divisor(diff_h, stride_y)
    stride_x = closest_divisor(diff_w, stride_x)
    return stride_x, stride_y


def benchmark_dataset(dataset, num_batches=20, batch_size=4):
    print(f"Benchmark: {num_batches} batchy po {batch_size} przykładów")
    start = time.time()
    total_examples = 0
    for i, (patches, targets) in enumerate(dataset.take(num_batches)):
        total_examples += patches.shape[0]
    time_spent = time.time() - start
    print(f"Czas: {time_spent:.2f} s")
    print(f"Przykłady / s: {total_examples / time_spent:.0f}")
    print(f"Batchy / s: {num_batches / time_spent:.2f}")


def plotTrainHistory(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history.history['loss'], label='train')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss function')
    axes[0].legend()

    axes[1].plot(history.history['val_loss'], label='val')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss function')
    axes[1].set_yscale('log')
    axes[1].legend()

    plt.subplots_adjust(wspace=0.3)
    os.makedirs('fig_png', exist_ok=True)
    plt.savefig('fig_png/train_history.png')


def plot_all_losses(history, save_path=None):
    loss_keys = [k for k in history.history.keys() if k.endswith('_loss') and not k.startswith('val_')]
    val_loss_keys = ['val_' + k for k in loss_keys]
    
    n = len(loss_keys)
    cols = 3
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten()
    
    for i, (loss_key, val_key) in enumerate(zip(loss_keys, val_loss_keys)):
        ax = axes[i]
        ax.plot(history.history[loss_key], label='train')
        ax.plot(history.history[val_key], label='val')
        ax.set_title(loss_key.replace('_loss', ''))
        ax.set_xlabel('Epoch')
        ax.legend()
        if i == 0:
            ax.set_ylabel('Loss')
    
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def prepare_metadata(df, clover_species, eps=0.001):
    """Dodaje zlogarytmowane zmienne docelowe oraz etykietę is_clover."""
    df = df.copy()
    target_cols = ['target_Dry_Green_g', 'target_Dry_Dead_g', 'target_Dry_Clover_g', 
                   'target_GDM_g', 'target_Dry_Total_g']
    for col in target_cols:
        df[f'log_{col}'] = np.log1p(df[col].values).astype(np.float32)
    
    if 'Height_Ave_cm' in df.columns:
        df['log_Height'] = np.log1p(df['Height_Ave_cm'].values).astype(np.float32)
    if 'Pre_GSHH_NDVI' in df.columns:
        df['NDVI'] = df['Pre_GSHH_NDVI'].values.astype(np.float32)
    
    df['is_clover'] = df['Species'].apply(lambda x: 1 if x in clover_species else 0).astype(np.float32)
    
    return df


def load_and_preprocess_image(path, is_training=False, augment=False):
    """Wczytuje obrazek, opcjonalnie zmienia rozmiar i normalizuje."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    if is_training and augment:
        img = apply_augmentations(img)
    norm_factor = 255.0 if tf.reduce_max(img) > 1 else 1.0
    img = img / norm_factor
    img = tf.image.rgb_to_hsv(img)
    return img


def extract_patches_from_image(img, patch_size=256, stride_x=158, stride_y=149):
    patches = tf.image.extract_patches(
        images=tf.expand_dims(img, 0),
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, stride_y, stride_x, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    return patches


def reshape_patches_flat(patches, patch_size):
    patches = tf.reshape(patches, [-1, patch_size, patch_size, 3])
    return patches


def reshape_patches_grid(patches, patch_size):
    shape = patches.shape.as_list()
    num_patches_y = shape[1]
    num_patches_x = shape[2]
    patches = tf.reshape(patches, (num_patches_y, num_patches_x, patch_size, patch_size, 3))
    return patches, num_patches_y, num_patches_x


def apply_augmentations(img):
    """Losowe przekształcenia dla jednej paczki."""
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img


def patch_generator_grid(df, image_dir, patch_size, stride_x, stride_y, is_training, augment):
    """Generuje pary (patch, targets) dla każdej paczki i każdego obrazu."""
    image_paths = df['image_path'].values
    
    for i in range(len(df)):
        img = load_and_preprocess_image(os.path.join(image_dir, image_paths[i]), is_training, augment)
        patches = extract_patches_from_image(img, patch_size, stride_x, stride_y)
        patches = tf.reshape(patches, [-1, patch_size, patch_size, 3])
        # targets
        targets = {
            'log_Dry_Green': df.loc[i, 'log_target_Dry_Green_g'], # df.loc[i, "target_Dry_Green_g"],
            'log_Dry_Dead': df.loc[i, 'log_target_Dry_Dead_g'], #df.loc[i, "target_Dry_Dead_g"], 
            'log_Dry_Clover': df.loc[i, 'log_target_Dry_Clover_g'], #df.loc[i, "target_Dry_Clover_g"], 
            'log_GDM': df.loc[i, 'log_target_GDM_g'], #df.loc[i, "target_GDM_g"], 
            'log_Total': df.loc[i, 'log_target_Dry_Total_g'], #df.loc[i, "target_Dry_Total_g"], 
        }
        if 'log_Height' in df.columns:
            targets['log_Height'] = df.loc[i, 'log_Height'] #df.loc[i, "Height_Ave_cm"]
        if 'NDVI' in df.columns:
            targets['NDVI'] = df.loc[i, 'NDVI']
        targets['is_clover'] = df.loc[i, 'is_clover']
        
        yield patches, targets


def get_patches_per_image(img_height, img_width, patch_size, stride_x, stride_y):
    n_y = (img_height - patch_size) // stride_y + 1 if img_height >= patch_size else 0
    n_x = (img_width  - patch_size) // stride_x + 1 if img_width  >= patch_size else 0
    return n_y * n_x


def patch_generator_flat(df, image_dir, patch_size, stride_x, stride_y):
    for i in range(len(df)):
        img = load_and_preprocess_image(os.path.join(image_dir, df.loc[i, 'image_path']))
        patches = extract_patches_from_image(img, patch_size, stride_x, stride_y)
        patches = reshape_patches_flat(patches, patch_size)
        for patch in patches:
            yield patch, patch  # autoenkoder: (x, x)


def create_ae_patch_dataset(df, image_dir, clover_species, batch_size=4, is_training=True, 
                         patch_size=256, stride_x=158, stride_y=149):
    """Tworzy Dataset z paczkami i celami."""
    df = prepare_metadata(df, clover_species) 
    
    # Jeden przykład w celu określenia sygnatury
    gen = patch_generator_flat(df, image_dir, patch_size, stride_x, stride_y)
    sample_patch, _ = next(gen)
    
    input_spec = tf.TensorSpec(shape=sample_patch.shape, dtype=sample_patch.dtype)
    
    dataset = tf.data.Dataset.from_generator(
        lambda: patch_generator_flat(df, image_dir, patch_size, stride_x, stride_y),
        output_signature=(input_spec, input_spec)
    )
    if is_training:
        img_height, img_width = 1000, 2000
        patches_per_image = get_patches_per_image(img_height, img_width, patch_size, stride_x, stride_y)
        buffer_size =  min(len(df) * patches_per_image // 20, 5000)
        dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_patch_dataset(df, image_dir, clover_species, batch_size=4, is_training=True, augment=True, 
                         patch_size=256, stride_x=158, stride_y=149):
    """Tworzy Dataset z paczkami i celami."""
    df = prepare_metadata(df, clover_species)
    
    # Jeden przykład w celu określenia sygnatury
    gen = patch_generator_grid(df, image_dir, patch_size, stride_x, stride_y, is_training, augment)
    sample_patches, sample_targets = next(gen)
    
    input_spec = tf.TensorSpec(shape=sample_patches.shape, dtype=sample_patches.dtype)
    target_spec = {k: tf.TensorSpec(shape=(), dtype=tf.float32) for k in sample_targets.keys()}
    
    dataset = tf.data.Dataset.from_generator(
        lambda: patch_generator_grid(df, image_dir, patch_size, stride_x, stride_y, is_training, augment),
        output_signature=(input_spec, target_spec)
    )
    if is_training:
        dataset = dataset.shuffle(buffer_size=100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


def inspect_images(df, image_dir, num_samples=20):
    sample_df = df.head(num_samples).copy()
    sample_df['image_path_full'] = sample_df['image_path'].apply(lambda x: f"{image_dir}/{x}")

    for idx, row in sample_df.iterrows():
        path = row['image_path_full']
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32)

        # Dane do wyświetlenia
        green = row.get('target_Dry_Green_g', 0)
        dead = row.get('target_Dry_Dead_g', 0)
        clover = row.get('target_Dry_Clover_g', 0)
        gdm = row.get('target_GDM_g', 0)
        total = row.get('target_Dry_Total_g', 0)
        state = row.get('State', "NA")
        month = pd.to_datetime(row.get('Sampling_Date', "1900/1/1"), errors='coerce').month if pd.notna(row.get('Sampling_Date', None)) else "NA"
        species = row.get('Species', "NA")

        # Tworzymy tabelę danych
        cell_text = [
            ["Green", f"{green:.2f}"],
            ["Dead", f"{dead:.2f}"],
            ["Clover", f"{clover:.2f}"],
            ["GDM", f"{gdm:.2f}"],
            ["Total", f"{total:.2f}"],
            ["State", str(state)],
            ["Month", str(month)],
            ["Species", str(species)],
        ]
        col_labels = ['Variable', 'Value']

        fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 2]})
        # Po lewej stronie obrazek
        ax_img = axs[0]
        ax_img.imshow(img.numpy() / 255.0)
        ax_img.set_title(f"Sample {idx}")
        ax_img.axis('off')

        # Po prawej stronie tabela bez nakładania się tekstu
        ax_table = axs[1]
        table = ax_table.table(cellText=cell_text, colLabels=col_labels, cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.8)
        ax_table.axis('off')
        plt.tight_layout()
        plt.show()


def show_reconstruction(model, dataset, num_samples=1):
    for images, _ in dataset.take(1):
        images = images[:num_samples]
        reconstructed = model.predict(images, verbose=0)
        reconstructed = np.clip(reconstructed, 0, 1)
        plt.figure(figsize=(2*num_samples, 4))
        for i in range(num_samples):
            plt.subplot(2, num_samples, i+1)
            plt.imshow(tf.image.hsv_to_rgb(images[i]).numpy())
            plt.title(f"Original")
            plt.axis('off')
            plt.subplot(2, num_samples, num_samples+i+1)
            plt.imshow(tf.image.hsv_to_rgb(reconstructed[i]))
            plt.title(f"Reconstructed")
            plt.axis('off')
        plt.tight_layout()
        plt.show()


import numpy as np
import matplotlib.pyplot as plt

def show_predictions(model, dataset, loss_name, mode="worst", num_examples=3, eps=0.001, num_cols=7):
    """
    Wyświetla patche (zgrupowane według obrazu) z największym błędem 
    dla wskazanej głowy (loss_name).
    """
    y_true_all = []
    y_pred_all = []
    patch_groups = []

    for patches, targets in dataset:
        preds = model.predict_on_batch(patches)
        y_pred = preds[loss_name].flatten()
        y_true = targets[loss_name].numpy().flatten()
        
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
        patch_groups.extend(patches.numpy())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    # Błąd MAE
    errors = np.abs(y_true_all - y_pred_all)
    if mode == "worst":
        top_indices = np.argsort(errors)[-num_examples:][::-1]
        mode_pl = "największym"
    elif mode == "best":
        top_indices = np.argsort(errors)[:num_examples]
        mode_pl = "najmnięszym"

    print(f"Top {num_examples} przykładów z {mode_pl} błędem MAE dla {loss_name}:")
    for idx in top_indices:
        print(f"\nPrzykład {idx}:")
        print(f"  Wartość rzeczywista (log): {y_true_all[idx]:.4f}")
        print(f"  Przewidywana (log): {y_pred_all[idx]:.4f}")
        print(f"  MAE: {errors[idx]:.6f}")
        
        true_original = np.expm1(y_true_all[idx]) if loss_name not in ['is_clover', 'NDVI'] else y_true_all[idx]
        pred_original = np.expm1(y_pred_all[idx]) if loss_name not in ['is_clover', 'NDVI'] else y_pred_all[idx]
        
        if loss_name not in ['is_clover', 'NDVI']:
            print(f"  Rzeczywista (oryginalna): {true_original:.2f}")
            print(f"  Przewidywana (oryginalna): {pred_original:.2f}")
        else:
            print(f"  Rzeczywista ({loss_name}): {true_original:.0f}")
            print(f"  Przewidywana ({loss_name}): {pred_original:.4f}")
        
        patches_group = patch_groups[idx]
        show_full_grid_from_patches(patches_group, title=f"Przykład {idx} (błąd: {errors[idx]:.4f})", cols=num_cols)


def show_full_grid_from_patches(patches, title="Patche", cols=5):
    num_patches = patches.shape[0]
    H, W, C = patches.shape[1], patches.shape[2], patches.shape[3]
    rows = (num_patches + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_patches):
        r = i // cols
        c = i % cols
        patch_rgb = tf.image.hsv_to_rgb(patches[i]).numpy()
        axes[r, c].imshow(patch_rgb)
        axes[r, c].axis('off')
        axes[r, c].set_title(f"{i}", fontsize=8)
    
    # puste podwykresy Wyłączone
    for i in range(num_patches, rows * cols):
        r = i // cols
        c = i % cols
        axes[r, c].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def weighted_r2_score(y_true, y_pred, weights):
    """
    Oblicza ważony współczynnik R².
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weights = np.array(weights)
    
    weighted_mean = np.average(y_true, weights=weights)
    weighted_mse = np.average((y_true - y_pred) ** 2, weights=weights)
    weighted_var = np.average((y_true - weighted_mean) ** 2, weights=weights)
    
    if weighted_var == 0:
        return 0.0
    return 1 - weighted_mse / weighted_var


def collect_predictions(model, dataset, model_heads, derived_heads, derived_formulas):
    """
    Zbiera przewidywania i wartości rzeczywiste dla wskazanych głów.
    """
    all_heads = model_heads + derived_heads
    y_true_dict = {name.replace("log_", ""): [] for name in all_heads}
    y_pred_dict = {name.replace("log_", ""): [] for name in all_heads}
    
    for patches, targets in dataset:
        preds = model.predict_on_batch(patches)
        
        # 1. Pobieramy przewidywania dla 3 głównych głów
        pred_orig = {}
        for name in model_heads:
            y_true = targets[name].numpy().flatten()
            y_pred = preds[name].flatten()
            
            # Konwersja z logarytmów na gramy
            y_true_orig = np.expm1(y_true)
            y_pred_orig = np.expm1(y_pred)
            
            orig_name = name.replace("log_", "")
            y_true_dict[orig_name].extend(y_true_orig)
            y_pred_dict[orig_name].extend(y_pred_orig)
            pred_orig[orig_name] = y_pred_orig
        
        # 2. Wyznaczamy zmienne pochodne (Clover i Dead)
        for dname in derived_heads:
            orig_name = dname.replace("log_", "")
            y_pred_derived = derived_formulas[orig_name](pred_orig)
            y_true_derived = np.expm1(targets[dname].numpy().flatten())
            
            y_true_dict[orig_name].extend(y_true_derived)
            y_pred_dict[orig_name].extend(y_pred_derived)
    
    return y_true_dict, y_pred_dict


def prepare_weighted_data(y_true_dict, y_pred_dict, head_list, weights_dict):
    all_y_true = []
    all_y_pred = []
    all_weights = []
    for head in head_list:
        head = head.replace("log_", "")
        all_y_true.extend(y_true_dict[head])
        all_y_pred.extend(y_pred_dict[head])
        weight = weights_dict[head]
        all_weights.extend([weight] * len(y_true_dict[head]))
    return all_y_true, all_y_pred, all_weights
