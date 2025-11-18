"""
Custom Convolutional Filters
Purpose: Aplica filtros de suavizado, detección de bordes y nitidez
Author: Tu Nombre
Date: 2025-01-17
"""

import tensorflow as tf
import numpy as np
from typing import Tuple

class ConvolutionFilters:
    """
    Clase para aplicar filtros convolucionales personalizados
    """
    
    def __init__(self):
        # Filtro 1: Suavizado (average blur)
        self.kernel_smooth = tf.constant([
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]
        ], dtype=tf.float32)
        
        # Filtro 2: Detección de bordes (Sobel)
        self.kernel_edge = tf.constant([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=tf.float32)
        
        # Filtro 3: Nitidez (sharpen)
        self.kernel_sharp = tf.constant([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=tf.float32)
    
    def apply_filters(self, images: np.ndarray) -> np.ndarray:
        """
        Aplica SOLO el filtro de suavizado a cada canal para mantener 3 canales
        """
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)
        
        images_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
        filtered_channels = []
        
        for i in range(images_tensor.shape[-1]):  # Para cada canal (R, G, B)
            channel = images_tensor[:, :, :, i:i+1]
            
            # Aplicar SOLO el filtro de suavizado (el más útil)
            smooth = tf.nn.conv2d(channel, self._expand_kernel(self.kernel_smooth), 
                                strides=[1, 1, 1, 1], padding='SAME')
            filtered_channels.append(smooth)
        
        # Concatenar los 3 canales filtrados
        result = tf.concat(filtered_channels, axis=-1)
        return result.numpy()

    
    def _expand_kernel(self, kernel: tf.Tensor) -> tf.Tensor:
        """Expande kernel para convolución 2D"""
        kernel = tf.expand_dims(kernel, axis=-1)  # Add input channels dimension
        kernel = tf.expand_dims(kernel, axis=-1)  # Add output channels dimension
        return kernel
    
    def get_filter_info(self) -> dict:
        """Retorna información sobre el filtro aplicado"""
        return {
            "filters_applied": [
                {
                    "name": "smoothing",
                    "description": "Filtro de promedio 3x3 - reduce ruido",
                    "kernel": self.kernel_smooth.numpy().tolist()
                }
            ],
            "note": "Se aplica únicamente el filtro de suavizado para mantener 3 canales de salida"
        }
