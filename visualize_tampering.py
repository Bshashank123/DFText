"""
VISUAL TAMPERING HEATMAP
Highlights suspicious regions in the image
IDEA 3 - Visual feedback for training/debugging
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from pipeline.preprocess import preprocess_image
from pipeline.image_forensics import (
    extract_noise_residual, noise_variance_map,
    extract_background_regions
)

class TamperingVisualizer:
    """Generate visual heatmaps of suspicious regions"""
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        self.gray = preprocess_image(self.img)
    
    def generate_noise_heatmap(self):
        """
        Highlight regions with suspicious noise patterns
        """
        noise = extract_noise_residual(self.gray)
        
        # Calculate local variance
        h, w = noise.shape
        block_size = 32
        heatmap = np.zeros((h, w))
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = noise[y:y+block_size, x:x+block_size]
                variance = np.var(block)
                
                # Suspicious if variance is too low (AI-like)
                if variance < 10:
                    suspicion = 1.0
                elif variance < 20:
                    suspicion = 0.5
                else:
                    suspicion = 0.0
                
                heatmap[y:y+block_size, x:x+block_size] = suspicion
        
        return heatmap
    
    def generate_stroke_heatmap(self):
        """
        Highlight regions with suspicious stroke patterns
        """
        # Get edges
        edges = cv2.Canny(self.gray, 50, 150)
        
        # Calculate local edge density
        kernel = np.ones((15, 15), np.float32) / 225
        edge_density = cv2.filter2D(edges.astype(float), -1, kernel)
        
        # Normalize to [0, 1]
        heatmap = edge_density / (edge_density.max() + 1e-6)
        
        # Invert: high density = real, low density = suspicious
        # But too perfect edges = also suspicious
        heatmap = 1.0 - heatmap
        
        return heatmap
    
    def generate_paper_heatmap(self):
        """
        Highlight regions with suspicious paper texture
        """
        h, w = self.gray.shape
        heatmap = np.zeros((h, w))
        
        block_size = 50
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = self.gray[y:y+block_size, x:x+block_size]
                
                # Check if this is background (paper)
                mean_intensity = np.mean(block)
                if mean_intensity > 180:  # Likely paper
                    variance = np.var(block)
                    
                    # Suspicious if too uniform
                    if variance < 30:
                        suspicion = 1.0
                    elif variance < 60:
                        suspicion = 0.5
                    else:
                        suspicion = 0.0
                    
                    heatmap[y:y+block_size, x:x+block_size] = suspicion
        
        return heatmap
    
    def generate_composite_heatmap(self):
        """
        Combine all heatmaps
        """
        noise_heat = self.generate_noise_heatmap()
        stroke_heat = self.generate_stroke_heatmap()
        paper_heat = self.generate_paper_heatmap()
        
        # Weighted combination
        composite = (
            0.40 * noise_heat +
            0.35 * stroke_heat +
            0.25 * paper_heat
        )
        
        # Normalize
        composite = (composite - composite.min()) / (composite.max() - composite.min() + 1e-6)
        
        return composite
    
    def visualize(self, output_path=None):
        """
        Create a visual report with heatmaps
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Custom colormap: green (safe) -> yellow -> red (suspicious)
        colors = ['green', 'yellow', 'red']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('suspicion', colors, N=n_bins)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Noise heatmap
        noise_heat = self.generate_noise_heatmap()
        im1 = axes[0, 1].imshow(noise_heat, cmap=cmap, vmin=0, vmax=1)
        axes[0, 1].set_title('Noise Pattern Suspicion', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Stroke heatmap
        stroke_heat = self.generate_stroke_heatmap()
        im2 = axes[0, 2].imshow(stroke_heat, cmap=cmap, vmin=0, vmax=1)
        axes[0, 2].set_title('Stroke Pattern Suspicion', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # Paper heatmap
        paper_heat = self.generate_paper_heatmap()
        im3 = axes[1, 0].imshow(paper_heat, cmap=cmap, vmin=0, vmax=1)
        axes[1, 0].set_title('Paper Texture Suspicion', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Composite heatmap
        composite = self.generate_composite_heatmap()
        im4 = axes[1, 1].imshow(composite, cmap=cmap, vmin=0, vmax=1)
        axes[1, 1].set_title('COMPOSITE SUSPICION MAP', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Overlay on original
        overlay = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB).copy()
        composite_resized = cv2.resize(composite, (overlay.shape[1], overlay.shape[0]))
        
        # Create red overlay for suspicious regions
        red_mask = (composite_resized > 0.5).astype(np.uint8) * 255
        overlay[:, :, 0] = np.maximum(overlay[:, :, 0], red_mask)
        
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Suspicious Regions Overlay', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Tampering Analysis: {self.image_path}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_suspicion_score(self):
        """
        Get overall suspicion score
        """
        composite = self.generate_composite_heatmap()
        
        # Percentage of highly suspicious pixels
        suspicious_ratio = np.mean(composite > 0.5)
        
        return suspicious_ratio


def main():
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_tampering.py <image_path> [output_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"Analyzing: {image_path}")
    
    viz = TamperingVisualizer(image_path)
    
    # Generate visualization
    if not output_path:
        output_path = image_path.replace('.', '_analysis.')
    
    viz.visualize(output_path)
    
    # Get suspicion score
    score = viz.get_suspicion_score()
    print(f"\nOverall Suspicion: {score*100:.1f}%")
    
    if score > 0.5:
        print("⚠️  HIGH SUSPICION - Possible tampering detected")
    elif score > 0.3:
        print("⚠️  MODERATE SUSPICION - Review recommended")
    else:
        print("✓ LOW SUSPICION - Appears authentic")


if __name__ == '__main__':
    main()