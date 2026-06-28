"""
main version 2
MICRONEEDLE DETECTOR v4 - TRIANGULAR NEEDLE OPTIMIZED - WITH MENU SYSTEM

Specialized for accurate measurement of triangular/conical microneedles
Uses 6 complementary methods for robust area calculation
Enhanced with menu system and analysis database
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import distance
from skimage import morphology
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Listbox, Scrollbar
import os
import time
import json
import shutil
from datetime import datetime


class InteractiveTriangleEditor:
    """Interactive triangle editor for precise needle area measurement."""
    
    def __init__(self, image, needles, stage="BEFORE"):
        self.image = image.copy()
        self.original_image = image.copy()
        self.stage = stage
        self.needles = [(int(x), int(y)) for x, y in needles]
        
        # Each needle gets 3 triangle vertices
        self.triangles = {}
        for x, y in self.needles:
            self.triangles[(x, y)] = [
                (x, y - 30), (x - 26, y + 15), (x + 26, y + 15)
            ]
        
        self.current_needle_idx = 0
        self.dragging_vertex = None
        self.window_name = f"Triangle Editor - {stage} Image"
        self.help_text = [
            "N: Next | P: Prev | R: Reset | A: Auto-fit | SPACE: Done | ESC: Cancel"
        ]
    
    def load_triangles(self, triangles_data):
        """Load pre-existing triangle data."""
        self.triangles = {}
        for needle_key, vertices in triangles_data.items():
            # Convert string key back to tuple
            if isinstance(needle_key, str):
                x, y = map(int, needle_key.strip('()').split(','))
                needle_key = (x, y)
            self.triangles[needle_key] = [tuple(v) for v in vertices]
        
    def get_current_needle(self):
        if 0 <= self.current_needle_idx < len(self.needles):
            return self.needles[self.current_needle_idx]
        return None
    
    def get_triangle(self, needle):
        return self.triangles.get(needle, [])
    
    def set_triangle_vertex(self, needle, vertex_idx, new_pos):
        if needle in self.triangles and 0 <= vertex_idx < 3:
            self.triangles[needle][vertex_idx] = new_pos
    
    def reset_triangle(self, needle):
        x, y = needle
        self.triangles[needle] = [(x, y - 30), (x - 26, y + 15), (x + 26, y + 15)]
    
    def auto_fit_triangle(self, needle):
        x, y = needle
        size = 50
        x1, y1 = max(0, x - size), max(0, y - size)
        x2, y2 = min(self.image.shape[1], x + size), min(self.image.shape[0], y + size)
        
        region = self.original_image[y1:y2, x1:x2]
        if len(region.shape) == 3:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        threshold = np.percentile(region, 80)
        mask = (region > threshold).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            
            if len(approx) >= 3:
                triangle = []
                for i in range(min(3, len(approx))):
                    px, py = approx[i][0][0] + x1, approx[i][0][1] + y1
                    triangle.append((px, py))
                
                triangle = sorted(triangle, key=lambda p: p[1])
                if len(triangle) >= 3:
                    top = triangle[0]
                    bottom = sorted(triangle[1:], key=lambda p: p[0])
                    self.triangles[needle] = [top] + bottom
                    return True
        return False
    
    def calculate_triangle_area(self, vertices):
        if len(vertices) != 3:
            return 0
        p1, p2, p3 = vertices
        return abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0
    
    def find_nearest_vertex(self, pos, threshold=15):
        needle = self.get_current_needle()
        if not needle:
            return None
        triangle = self.get_triangle(needle)
        for i, vertex in enumerate(triangle):
            dist = np.sqrt((pos[0] - vertex[0])**2 + (pos[1] - vertex[1])**2)
            if dist < threshold:
                return i
        return None
    
    def mouse_callback(self, event, x, y, flags, param):
        needle = self.get_current_needle()
        if not needle:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            vertex_idx = self.find_nearest_vertex((x, y))
            if vertex_idx is not None:
                self.dragging_vertex = vertex_idx
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_vertex is not None:
                self.set_triangle_vertex(needle, self.dragging_vertex, (x, y))
                self.update_display()
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_vertex = None
    
    def draw_display(self):
        display = self.image.copy()
        
        for i, needle in enumerate(self.needles):
            triangle = self.get_triangle(needle)
            pts = np.array(triangle, np.int32)
            
            if i == self.current_needle_idx:
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                for vx, vy in triangle:
                    cv2.circle(display, (vx, vy), 4, (0, 0, 255), -1)
                    cv2.circle(display, (vx, vy), 6, (255, 255, 255), 2)
            else:
                cv2.polylines(display, [pts], True, (100, 100, 100), 1)
            
            nx, ny = needle
            color = (0, 255, 255) if i == self.current_needle_idx else (150, 150, 150)
            cv2.circle(display, (nx, ny), 3, color, -1)
        
        needle = self.get_current_needle()
        if needle:
            triangle = self.get_triangle(needle)
            area = self.calculate_triangle_area(triangle)
            
            cv2.putText(display, f"{self.stage} - Needle {self.current_needle_idx + 1}/{len(self.needles)}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Area: {area:.1f} px²",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        for i, line in enumerate(self.help_text):
            cv2.putText(display, line, (10, display.shape[0] - 30 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return display
    
    def update_display(self):
        cv2.imshow(self.window_name, self.draw_display())
    
    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == 27:
                cv2.destroyWindow(self.window_name)
                return None
            elif key == ord('n'):
                self.current_needle_idx = (self.current_needle_idx + 1) % len(self.needles)
                self.update_display()
            elif key == ord('p'):
                self.current_needle_idx = (self.current_needle_idx - 1) % len(self.needles)
                self.update_display()
            elif key == ord('r'):
                needle = self.get_current_needle()
                if needle:
                    self.reset_triangle(needle)
                    self.update_display()
            elif key == ord('a'):
                needle = self.get_current_needle()
                if needle:
                    self.auto_fit_triangle(needle)
                    self.update_display()
        
        cv2.destroyWindow(self.window_name)
        return self.triangles


class MicroneedleDetector:
    
    def __init__(self):
        self.detected_needles = []
        self.database_folder = "dissolving test database"
        os.makedirs(self.database_folder, exist_ok=True)
        

    def detect_needles(self, image):
        """Detect needles with relaxed parameters - analyzes entire image."""
        print("\n=== DETECTING NEEDLES (ENTIRE IMAGE) ===")
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        print("  Finding local maxima...")
        coordinates = peak_local_max(
            blurred, 
            min_distance=22,
            threshold_abs=np.percentile(blurred, 92),
            exclude_border=20
        )
        
        maxima_points = [(int(c[1]), int(c[0])) for c in coordinates]
        print(f"    Found {len(maxima_points)} local maxima")
        
        print("  Applying morphological top-hat...")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
        
        threshold_value = np.percentile(tophat, 96)
        _, tophat_binary = cv2.threshold(tophat, threshold_value, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(tophat_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tophat_points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 3 < area < 200:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    tophat_points.append((cx, cy))
        
        print(f"    Found {len(tophat_points)} top-hat features")
        
        all_points = maxima_points + tophat_points
        return all_points
    
    def merge_close_points(self, points, min_distance=18):
        """Merge points that are very close together."""
        if not points:
            return []
        
        print(f"\n=== MERGING CLOSE DETECTIONS ===")
        print(f"  Initial points: {len(points)}")
        
        points = np.array(points)
        merged = []
        used = set()
        
        for i, point in enumerate(points):
            if i in used:
                continue
            
            distances = np.sqrt(np.sum((points - point)**2, axis=1))
            nearby_indices = np.where(distances < min_distance)[0]
            
            for idx in nearby_indices:
                used.add(int(idx))
            
            cluster = points[nearby_indices]
            centroid = np.mean(cluster, axis=0)
            merged.append((int(centroid[0]), int(centroid[1])))
        
        print(f"  After merging: {len(merged)} points")
        return merged
    
    def validate_needles(self, image, points):
        """Validate with shape and sharpness checks."""
        print("\n=== VALIDATING NEEDLES ===")
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        validated = []
        
        patch_median = np.median(gray)
        patch_std = np.std(gray)
        
        print(f"  Patch median: {patch_median:.1f}, std: {patch_std:.1f}")
        
        for x, y in points:
            if x < 15 or y < 15 or x >= gray.shape[1]-15 or y >= gray.shape[0]-15:
                continue
            
            patch = gray[y-8:y+8, x-8:x+8]
            if patch.size == 0:
                continue
            
            center = patch[6:10, 6:10]
            surround = patch.copy()
            surround[6:10, 6:10] = 0
            surround_masked = surround.flatten()
            
            if len(surround_masked) < 10:
                continue
            
            center_mean = np.mean(center)
            surround_mean = np.mean(surround_masked)
            
            if not (center_mean > surround_mean * 1.03 and center_mean > patch_median + patch_std * 0.3):
                continue
            
            larger_patch = gray[max(0, y-12):min(gray.shape[0], y+12), 
                               max(0, x-12):min(gray.shape[1], x+12)]
            
            if larger_patch.size < 100:
                continue
            
            sobelx = cv2.Sobel(larger_patch, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(larger_patch, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            max_gradient = np.max(gradient_magnitude)
            mean_gradient = np.mean(gradient_magnitude)
            
            if max_gradient < 30 or mean_gradient < 8:
                continue
            
            center_y, center_x = larger_patch.shape[0] // 2, larger_patch.shape[1] // 2
            
            gradient_directions = []
            for dy in [-6, -3, 0, 3, 6]:
                for dx in [-6, -3, 0, 3, 6]:
                    py, px = center_y + dy, center_x + dx
                    if 0 <= py < sobelx.shape[0] and 0 <= px < sobelx.shape[1]:
                        if sobelx[py, px]**2 + sobely[py, px]**2 > 100:
                            angle = np.arctan2(sobely[py, px], sobelx[py, px])
                            gradient_directions.append(angle)
            
            if len(gradient_directions) > 5:
                angles = np.array(gradient_directions)
                angle_std = np.std(angles)
                
                if angle_std < 0.5:
                    continue
            
            validated.append((x, y))
        
        print(f"  Validated: {len(validated)} needles")
        return validated
    
    def smart_grid_filtering(self, points, image, expected_count=97):
        """Smart filtering using grid pattern + brightness ranking."""
        print("\n=== SMART GRID FILTERING ===")
        
        if len(points) < 10:
            return points
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        points_array = np.array(points)
        
        all_distances = []
        for i in range(min(len(points_array), 300)):
            for j in range(i+1, min(len(points_array), 300)):
                dist = np.linalg.norm(points_array[i] - points_array[j])
                if 15 < dist < 120:
                    all_distances.append(dist)
        
        if not all_distances:
            return points
        
        hist, bins = np.histogram(all_distances, bins=40)
        typical_spacing = bins[np.argmax(hist)]
        
        print(f"  Typical spacing: {typical_spacing:.1f} pixels")
        print(f"  Current count: {len(points)} needles")
        
        if abs(len(points) - expected_count) < expected_count * 0.2:
            print(f"  Count is good, keeping all points")
            return points
        
        if len(points) > expected_count * 1.3:
            print(f"  Too many points, applying grid-aware filtering...")
            
            intensities = []
            for x, y in points:
                if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]:
                    patch = gray[max(0,y-3):min(gray.shape[0],y+3), 
                               max(0,x-3):min(gray.shape[1],x+3)]
                    intensities.append(np.mean(patch))
                else:
                    intensities.append(0)
            
            sorted_indices = np.argsort(intensities)[::-1]
            
            kept_points = []
            min_spacing = typical_spacing * 0.65
            
            for idx in sorted_indices:
                point = points_array[idx]
                
                too_close = False
                for kept_point in kept_points:
                    if np.linalg.norm(point - kept_point) < min_spacing:
                        too_close = True
                        break
                
                if not too_close:
                    kept_points.append(point)
                
                if len(kept_points) >= expected_count * 1.1:
                    break
            
            filtered = [(int(p[0]), int(p[1])) for p in kept_points]
            print(f"  After filtering: {len(filtered)} needles")
            return filtered
        
        print(f"  Count reasonable, no filtering")
        return points
    
    def refine_to_local_maxima(self, image, points):
        """Refine each point to nearest local maximum."""
        print("\n=== REFINING TO LOCAL MAXIMA ===")
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        refined = []
        
        for x, y in points:
            if x < 8 or y < 8 or x >= gray.shape[1]-8 or y >= gray.shape[0]-8:
                refined.append((x, y))
                continue
            
            patch = gray[y-7:y+7, x-7:x+7]
            
            if patch.size == 0:
                refined.append((x, y))
                continue
            
            max_loc = np.unravel_index(np.argmax(patch), patch.shape)
            refined_y = y - 7 + max_loc[0]
            refined_x = x - 7 + max_loc[1]
            
            refined.append((int(refined_x), int(refined_y)))
        
        print(f"  Refined {len(refined)} positions")
        return refined
    
    def final_deduplication(self, points, min_distance=20):
        """Final deduplication."""
        print("\n=== FINAL DEDUPLICATION ===")
        print(f"  Initial count: {len(points)}")
        
        if len(points) < 2:
            return points
        
        points_array = np.array(points)
        sorted_indices = np.lexsort((points_array[:, 0], points_array[:, 1]))
        sorted_points = points_array[sorted_indices]
        
        unique_points = []
        
        for point in sorted_points:
            is_duplicate = False
            for kept_point in unique_points:
                distance = np.linalg.norm(point - np.array(kept_point))
                if distance < min_distance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_points.append((int(point[0]), int(point[1])))
        
        print(f"  After deduplication: {len(unique_points)} unique needles")
        return unique_points
    
    def auto_remove_edge_outliers(self, points):
        """Asymmetric edge outlier removal."""
        print("\n=== AUTO EDGE OUTLIER REMOVAL (ASYMMETRIC) ===")
        print(f"  Initial count: {len(points)}")
        
        if len(points) < 20:
            print("  Too few points, skipping")
            return points
        
        points_array = np.array(points)
        x_coords = points_array[:, 0]
        y_coords = points_array[:, 1]
        
        x_lower = np.percentile(x_coords, 2)
        x_upper = np.percentile(x_coords, 88)
        
        y_lower = np.percentile(y_coords, 2)
        y_upper = np.percentile(y_coords, 88)
        
        print(f"  Asymmetric bounds:")
        print(f"    X: [{x_lower:.0f}, {x_upper:.0f}] (2nd-88th percentile)")
        print(f"    Y: [{y_lower:.0f}, {y_upper:.0f}] (2nd-88th percentile)")
        
        filtered_points = []
        removed_points = []
        
        for point in points:
            x, y = point
            
            if x_lower <= x <= x_upper and y_lower <= y <= y_upper:
                filtered_points.append(point)
            else:
                removed_points.append(point)
        
        print(f"  Removed {len(removed_points)} edge outliers")
        print(f"  Kept {len(filtered_points)} core needles")
        
        return filtered_points if filtered_points else points
    
    def filter_by_size_consistency(self, image, points):
        """Filter outliers based on needle size consistency."""
        print("\n=== SIZE CONSISTENCY FILTERING ===")
        print(f"  Initial count: {len(points)}")
        
        if len(points) < 10:
            print("  Too few points for size filtering, skipping")
            return points
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        sizes = []
        
        for x, y in points:
            if x < 20 or y < 20 or x >= gray.shape[1]-20 or y >= gray.shape[0]-20:
                continue
            
            patch = gray[y-15:y+15, x-15:x+15]
            
            if patch.size < 100:
                continue
            
            try:
                _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except:
                continue
            
            center_y, center_x = patch.shape[0] // 2, patch.shape[1] // 2
            
            mask = np.zeros((patch.shape[0]+2, patch.shape[1]+2), np.uint8)
            _, binary_filled, _, _ = cv2.floodFill(binary.copy(), mask, (center_x, center_y), 255)
            
            bright_area = np.sum(binary_filled > 0)
            
            contours, _ = cv2.findContours(binary_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter * perimeter)
                else:
                    circularity = 0
            else:
                circularity = 0
            
            sizes.append({
                'point': (x, y),
                'area': bright_area,
                'circularity': circularity
            })
        
        if len(sizes) < 5:
            print("  Not enough valid measurements, keeping all points")
            return points
        
        areas = np.array([s['area'] for s in sizes])
        circularities = np.array([s['circularity'] for s in sizes])
        
        median_area = np.median(areas)
        mad_area = np.median(np.abs(areas - median_area))
        
        print(f"  Median area: {median_area:.1f} pixels")
        print(f"  Median circularity: {np.median(circularities):.3f}")
        
        filtered_points = []
        removed_count = 0
        
        for size_info in sizes:
            area = size_info['area']
            circ = size_info['circularity']
            point = size_info['point']
            
            if mad_area > 0:
                area_deviation = abs(area - median_area) / mad_area
            else:
                area_deviation = 0
            
            if area_deviation < 3.5 and circ < 0.9:
                filtered_points.append(point)
            else:
                removed_count += 1
        
        print(f"  Removed {removed_count} outliers")
        print(f"  After size filtering: {len(filtered_points)} needles")
        
        return filtered_points if filtered_points else points
    
    def measure_triangular_needle(self, patch, stage="BEFORE", reference_data=None):
        """
        Measure triangular needle area using 6 complementary methods.
        Optimized for conical/triangular microneedle geometry.
        """
        if patch.size < 400:
            return 0, 0
        
        # Calculate patch statistics
        patch_mean = np.mean(patch)
        patch_std = np.std(patch)
        patch_max = np.max(patch)
        
        # For AFTER: check if needle is dissolved
        if reference_data is not None:
            min_intensity = reference_data['min_needle_intensity']
            min_peak = reference_data['min_peak_intensity']
            
            # If patch is too dark or no bright peak, needle is dissolved
            if patch_mean < min_intensity or patch_max < min_peak:
                return 0, patch_mean
        
        # METHOD 1: Adaptive Otsu Thresholding
        try:
            _, otsu_binary = cv2.threshold(patch.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_area = np.sum(otsu_binary > 0)
        except:
            otsu_area = 0
        
        # METHOD 2: Gradient-Based Edge Detection
        sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        edge_threshold = np.percentile(gradient_magnitude, 80)
        edges = gradient_magnitude > edge_threshold
        edges_filled = ndimage.binary_fill_holes(edges)
        gradient_area = np.sum(edges_filled)
        
        # METHOD 3: Intensity-Weighted Area
        intensity_threshold = patch_mean + 0.25 * patch_std
        above_threshold = patch > intensity_threshold
        
        patch_normalized = (patch - patch.min()) / (patch.max() - patch.min() + 1e-6)
        if np.sum(patch_normalized > 0.25) > 0:
            weighted_area = np.sum(patch_normalized[above_threshold]) * np.sum(above_threshold) / np.sum(patch_normalized > 0.25)
        else:
            weighted_area = 0
        
        # METHOD 4: Contour-Based Measurement
        high_threshold = patch_mean + 0.4 * patch_std
        _, binary_high = cv2.threshold(patch.astype(np.uint8), int(high_threshold), 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3, 3), np.uint8)
        binary_clean = cv2.morphologyEx(binary_high, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
        else:
            contour_area = 0
            hull_area = 0
        
        # METHOD 5: Peak-Based Triangular Measurement
        center_y, center_x = patch.shape[0] // 2, patch.shape[1] // 2
        peak_region = patch[max(0,center_y-5):min(patch.shape[0],center_y+5), 
                           max(0,center_x-5):min(patch.shape[1],center_x+5)]
        
        if peak_region.size > 0:
            peak_intensity = np.max(peak_region)
            # Triangular needles: keep pixels at least 40% of peak brightness
            decay_threshold = peak_intensity * 0.4
            triangular_mask = patch > decay_threshold
            triangular_area = np.sum(triangular_mask)
        else:
            triangular_area = 0
            peak_intensity = patch_max
        
        # METHOD 6: Adaptive Local Thresholding
        try:
            adaptive_binary = cv2.adaptiveThreshold(
                patch.astype(np.uint8), 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, -5
            )
            adaptive_area = np.sum(adaptive_binary > 0)
        except:
            adaptive_area = 0
        
        # COMBINE METHODS: Weighted average favoring methods that work well for triangular shapes
        areas = [
            (otsu_area, 0.12),          # Otsu thresholding
            (gradient_area, 0.20),      # Edge-based (good for triangular boundaries)
            (weighted_area, 0.12),      # Intensity-weighted
            (contour_area, 0.18),       # Contour-based
            (hull_area, 0.08),          # Convex hull
            (triangular_area, 0.25),    # Triangle-specific (highest weight)
            (adaptive_area, 0.05)       # Adaptive threshold
        ]
        
        # Filter out zero/invalid areas and calculate weighted average
        valid_areas = [(a, w) for a, w in areas if a > 0 and a < patch.size]
        
        if valid_areas:
            total_weight = sum(w for _, w in valid_areas)
            final_area = sum(a * w for a, w in valid_areas) / total_weight
        else:
            final_area = 0
        
        # Calculate average intensity for the needle region
        if final_area > 50:  # Only if we detected significant needle area
            if otsu_area > 0:
                needle_pixels = patch[otsu_binary > 0]
                avg_intensity = np.mean(needle_pixels) if len(needle_pixels) > 0 else patch_mean
            else:
                avg_intensity = patch_mean
        else:
            avg_intensity = patch_mean
        
        return final_area, avg_intensity
    
    def calculate_needle_areas(self, image, positions, stage="BEFORE", reference_thresholds=None):
        """FIXED area calculation - properly detects BRIGHT needle material."""
        print(f"\n=== CALCULATING NEEDLE AREAS ({stage}) - FIXED METHOD ===")
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        total_area = 0
        total_intensity = 0
        needle_count = len(positions)
        active_needles = 0
        dissolved_needles = 0
        
        # Store individual needle thresholds for consistency
        if stage == "BEFORE":
            reference_thresholds = {
                'needle_thresholds': {},
                'needle_areas': {}
            }
        
        print(f"  Processing {needle_count} needles...")
        
        for idx, (x, y) in enumerate(positions):
            if x < 20 or y < 20 or x >= gray.shape[1]-20 or y >= gray.shape[0]-20:
                continue
            
            # Use 40x40 patch for better coverage
            patch = gray[y-20:y+20, x-20:x+20]
            if patch.size == 0:
                continue
            
            # ===========================================
            # FOCUS ON BRIGHT AREAS (NEEDLE MATERIAL)
            # ===========================================
            
            # Get patch statistics
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)
            patch_max = np.max(patch)
            patch_min = np.min(patch)
            
            # Find the bright pixels (needle material is BRIGHT/WHITE)
            # Use percentile to find bright threshold
            bright_percentile = 75  # Top 25% brightest pixels
            bright_threshold = np.percentile(patch, bright_percentile)
            
            # For very bright patches, be more aggressive
            if patch_max > 230:  # Very bright needle
                bright_threshold = max(bright_threshold, patch_mean + patch_std)
            
            # Create binary mask of bright areas
            _, bright_mask = cv2.threshold(patch, int(bright_threshold), 255, cv2.THRESH_BINARY)
            
            # Clean up the mask - remove noise
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel_small)
            bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_small)
            
            # Find the largest connected component (main needle area)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
            
            if num_labels > 1:  # At least one component besides background
                # Find the largest component (excluding background at label 0)
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_idx = np.argmax(areas) + 1
                
                # Create mask with only the largest component
                needle_mask = np.zeros_like(labels, dtype=np.uint8)
                needle_mask[labels == largest_idx] = 255
                
                # Calculate area of the needle
                needle_area = np.sum(needle_mask > 0)
            else:
                # No bright areas found - might be dissolved
                needle_area = 0
            
            # ===========================================
            # STAGE-SPECIFIC PROCESSING
            # ===========================================
            
            if stage == "BEFORE":
                # Store the threshold and area for this needle
                reference_thresholds['needle_thresholds'][idx] = bright_threshold
                reference_thresholds['needle_areas'][idx] = needle_area
                
                # Validate the area is reasonable for a needle
                if needle_area < 10:  # Too small, try alternative method
                    # Try Otsu's method as backup
                    otsu_thresh, binary_otsu = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Make sure Otsu threshold is selecting bright areas
                    if otsu_thresh < patch_mean:
                        # Otsu selected dark areas, invert
                        binary_otsu = cv2.bitwise_not(binary_otsu)
                    
                    # Clean and measure
                    binary_otsu = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, kernel_small)
                    binary_otsu = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel_small)
                    
                    alternative_area = np.sum(binary_otsu > 0)
                    if alternative_area > needle_area:
                        needle_area = alternative_area
                        reference_thresholds['needle_thresholds'][idx] = otsu_thresh
            
            elif stage == "AFTER" and reference_thresholds:
                # Use the same threshold from BEFORE for consistency
                if idx in reference_thresholds.get('needle_thresholds', {}):
                    before_threshold = reference_thresholds['needle_thresholds'][idx]
                    before_area = reference_thresholds['needle_areas'][idx]
                    
                    # Apply the same threshold
                    _, after_mask = cv2.threshold(patch, int(before_threshold), 255, cv2.THRESH_BINARY)
                    
                    # Clean up
                    after_mask = cv2.morphologyEx(after_mask, cv2.MORPH_OPEN, kernel_small)
                    after_mask = cv2.morphologyEx(after_mask, cv2.MORPH_CLOSE, kernel_small)
                    
                    # Measure area
                    measured_area = np.sum(after_mask > 0)
                    
                    # If area increased (shouldn't happen), use current bright detection
                    if measured_area > before_area * 1.1:
                        # Fall back to current bright detection
                        needle_area = needle_area  # Keep the current calculation
                    else:
                        needle_area = measured_area
            
            # ===========================================
            # DISSOLUTION STATUS CHECK
            # ===========================================
            
            # Check if needle is dissolved
            # A dissolved needle has very little or no bright material
            if stage == "AFTER" and reference_thresholds:
                if idx in reference_thresholds.get('needle_areas', {}):
                    before_area = reference_thresholds['needle_areas'][idx]
                    if before_area > 0:
                        dissolution_ratio = needle_area / before_area
                        if dissolution_ratio < 0.1:  # Less than 10% of original area
                            dissolved_needles += 1
                        else:
                            active_needles += 1
                    else:
                        if needle_area < 10:
                            dissolved_needles += 1
                        else:
                            active_needles += 1
                else:
                    if needle_area < 10:
                        dissolved_needles += 1
                    else:
                        active_needles += 1
            else:
                # For BEFORE image, all needles with area > 10 are active
                if needle_area < 10:
                    dissolved_needles += 1
                else:
                    active_needles += 1
            
            total_area += needle_area
            
            # Calculate intensity for this needle
            if needle_area > 0:
                # Use the bright pixels for intensity calculation
                bright_pixels = patch[bright_mask > 0] if 'bright_mask' in locals() else patch[patch > bright_threshold]
                if len(bright_pixels) > 0:
                    needle_intensity = np.mean(bright_pixels)
                else:
                    needle_intensity = patch_mean
                total_intensity += needle_intensity
        
        avg_intensity = total_intensity / max(needle_count, 1)
        
        print(f"\n  === Area Calculation Summary ===")
        print(f"  Total needles: {needle_count}")
        print(f"  Active needles (with material): {active_needles}")
        print(f"  Dissolved needles (no material): {dissolved_needles}")
        print(f"  Total area: {total_area:.2f} pixels²")
        print(f"  Average area per needle: {total_area/max(needle_count,1):.2f} pixels²")
        print(f"  Average intensity: {avg_intensity:.1f}/255")
        
        result = {
            'num_needles': needle_count,
            'active_needles': active_needles,
            'dissolved_needles': dissolved_needles,
            'total_area': total_area,
            'avg_intensity': avg_intensity,
            'reference_thresholds': reference_thresholds if stage == "BEFORE" else None
        }
        
        return result
    
    def calculate_triangle_areas(self, image, triangles, stage=""):
        """Calculate areas from user-defined triangles."""
        print(f"\n=== CALCULATING TRIANGLE AREAS ({stage}) ===")
        
        total_area = 0
        needle_areas = []
        
        for needle, vertices in triangles.items():
            if len(vertices) != 3:
                continue
            
            p1, p2, p3 = vertices
            area = abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0
            total_area += area
            needle_areas.append(area)
        
        # Calculate intensity
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        mask = np.zeros_like(gray)
        for vertices in triangles.values():
            pts = np.array(vertices, np.int32)
            cv2.fillPoly(mask, [pts], 255)
        
        avg_intensity = np.mean(gray[mask > 0]) if np.any(mask > 0) else 0
        
        print(f"  Total area: {total_area:.2f} pixels²")
        print(f"  Number of needles: {len(triangles)}")
        print(f"  Average needle area: {total_area/len(triangles):.2f} pixels²")
        print(f"  Average intensity: {avg_intensity:.1f}/255")
        
        return {
            'total_area': total_area,
            'num_needles': len(triangles),
            'needle_areas': needle_areas,
            'avg_intensity': avg_intensity,
            'active_needles': len(triangles),
            'reference_thresholds': None
        }
    
    def visualize_triangles(self, image, triangles, stage):
        """Visualize triangles on image."""
        output = image.copy()
        
        for needle, vertices in triangles.items():
            pts = np.array(vertices, np.int32)
            
            overlay = output.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)
            
            cv2.polylines(output, [pts], True, (0, 255, 255), 2)
            
            for vx, vy in vertices:
                cv2.circle(output, (vx, vy), 4, (0, 0, 255), -1)
            
            nx, ny = needle
            cv2.circle(output, (nx, ny), 3, (255, 0, 255), -1)
        
        cv2.putText(output, f"{stage}: {len(triangles)} triangles", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return output
    
    def interactive_manual_selection(self, image, points):
        """Interactive manual selection."""
        print("\n=== INTERACTIVE MANUAL SELECTION ===")
        print("  INSTRUCTIONS:")
        print("  - LEFT CLICK on dot: Remove it")
        print("  - LEFT CLICK on empty: Add new dot")
        print("  - RIGHT CLICK: Undo")
        print("  - ENTER: Accept | ESC: Skip | 'r': Reset")
        
        max_display_size = 1200
        h, w = image.shape[:2]
        scale = 1.0
        
        if w > max_display_size or h > max_display_size:
            scale = min(max_display_size / w, max_display_size / h)
            display_w = int(w * scale)
            display_h = int(h * scale)
            display_img = cv2.resize(image, (display_w, display_h))
            print(f"  Scaling: {scale:.2f}x ({w}x{h} -> {display_w}x{display_h})")
        else:
            display_img = image.copy()
        
        current_points = list(points)
        original_points = list(points)
        removed_points = []
        added_points = []
        action_history = []
        
        click_radius = max(10, int(10 * scale))
        
        def find_nearest_point(x, y):
            min_dist = float('inf')
            nearest_idx = -1
            
            for idx, point in enumerate(current_points):
                scaled_x = int(point[0] * scale)
                scaled_y = int(point[1] * scale)
                dist = np.sqrt((x - scaled_x)**2 + (y - scaled_y)**2)
                
                if dist < click_radius and dist < min_dist:
                    min_dist = dist
                    nearest_idx = idx
            
            return nearest_idx if min_dist < click_radius else -1
        
        def draw_points():
            temp_img = display_img.copy()
            
            for point in current_points:
                scaled_x = int(point[0] * scale)
                scaled_y = int(point[1] * scale)
                
                color = (0, 255, 0) if point in added_points else (0, 255, 255)
                cv2.circle(temp_img, (scaled_x, scaled_y), max(3, int(4 * scale)), color, -1)
            
            for point in removed_points:
                scaled_x = int(point[0] * scale)
                scaled_y = int(point[1] * scale)
                size = max(8, int(10 * scale))
                cv2.line(temp_img, (scaled_x-size, scaled_y-size), (scaled_x+size, scaled_y+size), (0, 0, 255), 2)
                cv2.line(temp_img, (scaled_x+size, scaled_y-size), (scaled_x-size, scaled_y+size), (0, 0, 255), 2)
            
            text = f"Current: {len(current_points)} | Removed: {len(removed_points)} | Added: {len(added_points)}"
            cv2.putText(temp_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return temp_img
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_points, removed_points, added_points, action_history
            
            if event == cv2.EVENT_LBUTTONDOWN:
                nearest_idx = find_nearest_point(x, y)
                
                if nearest_idx >= 0:
                    removed_point = current_points[nearest_idx]
                    current_points.pop(nearest_idx)
                    removed_points.append(removed_point)
                    action_history.append(('remove', removed_point))
                    print(f"  Removed: {removed_point}")
                else:
                    new_x = int(x / scale)
                    new_y = int(y / scale)
                    new_point = (new_x, new_y)
                    current_points.append(new_point)
                    added_points.append(new_point)
                    action_history.append(('add', new_point))
                    print(f"  Added: {new_point}")
                
                temp_img = draw_points()
                cv2.imshow('Manual Selection', temp_img)
            
            elif event == cv2.EVENT_RBUTTONDOWN:
                if action_history:
                    action_type, point = action_history.pop()
                    
                    if action_type == 'remove':
                        current_points.append(point)
                        removed_points.remove(point)
                        print(f"  Undo remove: {point}")
                    elif action_type == 'add':
                        current_points.remove(point)
                        added_points.remove(point)
                        print(f"  Undo add: {point}")
                    
                    temp_img = draw_points()
                    cv2.imshow('Manual Selection', temp_img)
        
        cv2.namedWindow('Manual Selection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Manual Selection', mouse_callback)
        
        temp_img = draw_points()
        cv2.imshow('Manual Selection', temp_img)
        
        done = False
        skip = False
        
        while not done and not skip:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:
                done = True
            elif key == 27:
                skip = True
            elif key == ord('r'):
                current_points = list(original_points)
                removed_points = []
                added_points = []
                action_history = []
                print("  Reset to original")
                temp_img = draw_points()
                cv2.imshow('Manual Selection', temp_img)
        
        cv2.destroyAllWindows()
        
        if skip:
            print("  Skipped")
            return points
        
        print(f"\n  Final count: {len(current_points)}")
        print(f"  Removed: {len(removed_points)} | Added: {len(added_points)}")
        
        return current_points
    
    def visualize_results(self, image, needle_points, stage):
        """Visualize detected needles."""
        output = image.copy()
        
        if len(needle_points) >= 3:
            points_array = np.array(needle_points)
            hull = cv2.convexHull(points_array.astype(np.float32))
            cv2.drawContours(output, [hull.astype(np.int32)], -1, (255, 0, 255), 2)
        
        for x, y in needle_points:
            cv2.circle(output, (x, y), 6, (0, 0, 255), 1)
            cv2.circle(output, (x, y), 2, (0, 255, 255), -1)
        
        text = f"{stage}: {len(needle_points)} needles"
        cv2.putText(output, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return output

    def save_analysis(self, analysis_name, before_image_path, after_image_path,
                     needles_before, needles_after, triangles_before, triangles_after,
                     before_data, after_data, reference_thresholds=None):
        """Save complete analysis to database."""
        analysis_folder = os.path.join(self.database_folder, analysis_name)
        os.makedirs(analysis_folder, exist_ok=True)
        
        # Copy images to analysis folder (only if not already there)
        before_img_dest = os.path.join(analysis_folder, "before_image.png")
        after_img_dest = os.path.join(analysis_folder, "after_image.png")
        
        # Convert paths to absolute paths for comparison
        before_src_abs = os.path.abspath(before_image_path)
        before_dst_abs = os.path.abspath(before_img_dest)
        after_src_abs = os.path.abspath(after_image_path)
        after_dst_abs = os.path.abspath(after_img_dest)
        
        # Only copy if source and destination are different
        if before_src_abs != before_dst_abs:
            shutil.copy(before_image_path, before_img_dest)
        
        if after_src_abs != after_dst_abs:
            shutil.copy(after_image_path, after_img_dest)
        
        # Convert triangles dict keys to strings (JSON doesn't support tuple keys)
        def triangles_to_serializable(triangles):
            if triangles is None:
                return None
            return {str(k): v for k, v in triangles.items()}
        
        # Save analysis data
        analysis_data = {
            'analysis_name': analysis_name,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'before_image': before_img_dest,
            'after_image': after_img_dest,
            'needles_before': needles_before,
            'needles_after': needles_after,
            'triangles_before': triangles_to_serializable(triangles_before),
            'triangles_after': triangles_to_serializable(triangles_after),
            'before_data': before_data,
            'after_data': after_data,
            'reference_thresholds': reference_thresholds
        }
        
        with open(os.path.join(analysis_folder, "analysis_data.json"), 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"\n✓ Analysis saved to: {analysis_folder}")
    
    def load_analysis(self, analysis_name):
        """Load analysis from database."""
        analysis_folder = os.path.join(self.database_folder, analysis_name)
        data_file = os.path.join(analysis_folder, "analysis_data.json")
        
        if not os.path.exists(data_file):
            raise ValueError(f"Analysis not found: {analysis_name}")
        
        with open(data_file, 'r') as f:
            analysis_data = json.load(f)
        
        print(f"\n✓ Loaded analysis: {analysis_name}")
        return analysis_data
    
    def list_analyses(self):
        """List all saved analyses."""
        if not os.path.exists(self.database_folder):
            return []
        
        analyses = []
        for item in os.listdir(self.database_folder):
            item_path = os.path.join(self.database_folder, item)
            if os.path.isdir(item_path):
                data_file = os.path.join(item_path, "analysis_data.json")
                if os.path.exists(data_file):
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                        analyses.append({
                            'name': item,
                            'date': data.get('date', 'Unknown'),
                            'data': data
                        })
        
        return sorted(analyses, key=lambda x: x['date'], reverse=True)

    def process_new_analysis(self, analysis_name, before_image_path, after_image_path):
        """Process a new analysis from scratch."""
        print(f"\n{'='*60}")
        print(f"NEW ANALYSIS: {analysis_name}")
        print(f"{'='*60}")
        
        # Process BEFORE image
        print(f"\n{'='*60}")
        print("STAGE 1: BEFORE IMAGE - Detection + Area Measurement")
        print(f"{'='*60}")
        
        image_before = cv2.imread(before_image_path)
        if image_before is None:
            raise ValueError(f"Cannot read image: {before_image_path}")
        
        print(f"Image: {os.path.basename(before_image_path)}")
        print(f"Size: {image_before.shape[1]}x{image_before.shape[0]} pixels")
        
        # Full detection pipeline
        detections = self.detect_needles(image_before)
        merged = self.merge_close_points(detections, min_distance=18)
        validated = self.validate_needles(image_before, merged)
        filtered = self.smart_grid_filtering(validated, image_before, expected_count=97)
        refined = self.refine_to_local_maxima(image_before, filtered)
        deduplicated = self.final_deduplication(refined, min_distance=20)
        size_filtered = self.filter_by_size_consistency(image_before, deduplicated)
        auto_filtered = self.auto_remove_edge_outliers(size_filtered)
        
        # Interactive selection
        needles_before = self.interactive_manual_selection(image_before, auto_filtered)
        
        # Triangle editor for BEFORE
        print("\n=== INTERACTIVE TRIANGLE ADJUSTMENT (BEFORE) ===")
        print("  Opening triangle editor... Adjust triangles to match needle boundaries")
        editor = InteractiveTriangleEditor(image_before, needles_before, stage="BEFORE")
        triangles_before = editor.run()
        
        # FALLBACK LOGIC: Use original method if triangles cancelled
        if triangles_before is None:
            print("  Triangle editing cancelled, using original method")
            before_data = self.calculate_needle_areas(image_before, needles_before, stage="BEFORE")
            triangles_before = None
        else:
            before_data = self.calculate_triangle_areas(image_before, triangles_before, stage="BEFORE")
        
        # Extract reference thresholds (only exists in original method, not triangle mode)
        reference_thresholds = before_data.get('reference_thresholds', None) if before_data else None
        
        # Process AFTER image
        print(f"\n{'='*60}")
        print("STAGE 2: AFTER IMAGE - Detection + Area Measurement")
        print(f"{'='*60}")
        
        image_after = cv2.imread(after_image_path)
        if image_after is None:
            raise ValueError(f"Cannot read image: {after_image_path}")
        
        print(f"Image: {os.path.basename(after_image_path)}")
        print(f"Size: {image_after.shape[1]}x{image_after.shape[0]} pixels")
        
        # Full detection pipeline for AFTER
        detections = self.detect_needles(image_after)
        merged = self.merge_close_points(detections, min_distance=18)
        validated = self.validate_needles(image_after, merged)
        filtered = self.smart_grid_filtering(validated, image_after, expected_count=97)
        refined = self.refine_to_local_maxima(image_after, filtered)
        deduplicated = self.final_deduplication(refined, min_distance=20)
        size_filtered = self.filter_by_size_consistency(image_after, deduplicated)
        auto_filtered = self.auto_remove_edge_outliers(size_filtered)
        
        # Interactive selection for AFTER
        needles_after = self.interactive_manual_selection(image_after, auto_filtered)
        
        # Triangle editor for AFTER
        print("\n=== INTERACTIVE TRIANGLE ADJUSTMENT (AFTER) ===")
        print("  Opening triangle editor... Adjust NEW triangles for AFTER image")
        editor_after = InteractiveTriangleEditor(image_after, needles_after, stage="AFTER")
        triangles_after = editor_after.run()
        
        # FALLBACK LOGIC: Use original method if triangles cancelled
        if triangles_after is None:
            print("  Triangle editing cancelled, using original method")
            after_data = self.calculate_needle_areas(image_after, needles_after, stage="AFTER", reference_thresholds=reference_thresholds)
            triangles_after = None
        else:
            after_data = self.calculate_triangle_areas(image_after, triangles_after, stage="AFTER")
        
        # Save analysis
        self.save_analysis(analysis_name, before_image_path, after_image_path,
                          needles_before, needles_after, triangles_before, triangles_after,
                          before_data, after_data, reference_thresholds)
        
        # Show results
        self.show_results(image_before, image_after, needles_before, needles_after,
                         triangles_before, triangles_after, before_data, after_data)
        
        return True
    
    def process_previous_analysis(self, analysis_name):
        """Re-process a previous analysis with ability to edit."""
        analysis_data = self.load_analysis(analysis_name)
        
        print(f"\n{'='*60}")
        print(f"RE-EDITING ANALYSIS: {analysis_name}")
        print(f"Date: {analysis_data['date']}")
        print(f"{'='*60}")
        
        # Load images
        image_before = cv2.imread(analysis_data['before_image'])
        image_after = cv2.imread(analysis_data['after_image'])
        
        if image_before is None or image_after is None:
            raise ValueError("Cannot load images from saved analysis")
        
        # Load saved data
        needles_before = [tuple(p) for p in analysis_data['needles_before']]
        needles_after = [tuple(p) for p in analysis_data['needles_after']]
        reference_thresholds = analysis_data.get('reference_thresholds', None)
        
        # Check if original used triangles or not
        original_had_triangles = analysis_data.get('triangles_before') is not None
        
        if original_had_triangles:
            # RE-EDIT BEFORE triangles
            print("\n=== RE-EDITING BEFORE TRIANGLES ===")
            editor_before = InteractiveTriangleEditor(image_before, needles_before, stage="BEFORE")
            
            if analysis_data['triangles_before'] is not None:
                editor_before.load_triangles(analysis_data['triangles_before'])
            
            triangles_before = editor_before.run()
            
            if triangles_before is None:
                print("  Triangle editing cancelled, using original method")
                before_data = self.calculate_needle_areas(image_before, needles_before, stage="BEFORE")
                triangles_before = None
            else:
                before_data = self.calculate_triangle_areas(image_before, triangles_before, stage="BEFORE")
            
            # Extract reference thresholds
            reference_thresholds = before_data.get('reference_thresholds', None)
            
            # RE-EDIT AFTER triangles
            print("\n=== RE-EDITING AFTER TRIANGLES ===")
            editor_after = InteractiveTriangleEditor(image_after, needles_after, stage="AFTER")
            
            if analysis_data['triangles_after'] is not None:
                editor_after.load_triangles(analysis_data['triangles_after'])
            
            triangles_after = editor_after.run()
            
            if triangles_after is None:
                print("  Triangle editing cancelled, using original method")
                after_data = self.calculate_needle_areas(image_after, needles_after, stage="AFTER", reference_thresholds=reference_thresholds)
                triangles_after = None
            else:
                after_data = self.calculate_triangle_areas(image_after, triangles_after, stage="AFTER")
        
        else:
            # Original used automatic method - offer to use triangles or keep automatic
            print("\n  Original analysis used automatic method (no triangles)")
            print("  Opening triangle editor... You can now add triangles or skip (ESC) to keep automatic method")
            
            # BEFORE
            print("\n=== TRIANGLE EDITOR FOR BEFORE ===")
            editor_before = InteractiveTriangleEditor(image_before, needles_before, stage="BEFORE")
            triangles_before = editor_before.run()
            
            if triangles_before is None:
                print("  Keeping original automatic method")
                before_data = self.calculate_needle_areas(image_before, needles_before, stage="BEFORE")
            else:
                before_data = self.calculate_triangle_areas(image_before, triangles_before, stage="BEFORE")
            
            reference_thresholds = before_data.get('reference_thresholds', reference_thresholds)
            
            # AFTER
            print("\n=== TRIANGLE EDITOR FOR AFTER ===")
            editor_after = InteractiveTriangleEditor(image_after, needles_after, stage="AFTER")
            triangles_after = editor_after.run()
            
            if triangles_after is None:
                print("  Keeping original automatic method")
                after_data = self.calculate_needle_areas(image_after, needles_after, stage="AFTER", reference_thresholds=reference_thresholds)
            else:
                after_data = self.calculate_triangle_areas(image_after, triangles_after, stage="AFTER")
        
        # Save updated analysis
        before_image_path = analysis_data['before_image']
        after_image_path = analysis_data['after_image']
        
        self.save_analysis(analysis_name, before_image_path, after_image_path,
                          needles_before, needles_after, triangles_before, triangles_after,
                          before_data, after_data, reference_thresholds)
        
        # Show results
        self.show_results(image_before, image_after, needles_before, needles_after,
                         triangles_before, triangles_after, before_data, after_data)
    
    def show_results(self, image_before, image_after, needles_before, needles_after,
                    triangles_before, triangles_after, before_data, after_data):
        """Display analysis results."""
        print("\n" + "="*60)
        print("DISSOLUTION ANALYSIS RESULTS")
        print("="*60)
        
        area_before = before_data['total_area']
        area_after = after_data['total_area']
        area_lost = area_before - area_after
        dissolution_percent = (area_lost / area_before) * 100 if area_before > 0 else 0
        
        intensity_before = before_data['avg_intensity']
        intensity_after = after_data['avg_intensity']
        intensity_drop = intensity_before - intensity_after
        
        needles_count_before = before_data.get('active_needles', before_data['num_needles'])
        needles_count_after = after_data.get('active_needles', after_data['num_needles'])
        
        print(f"\nBEFORE:")
        print(f"  Total Area: {area_before:.2f} pixels² ({area_before/1000:.4f} mm²)")
        print(f"  Avg Intensity: {intensity_before:.1f}/255")
        print(f"  Needles with material: {needles_count_before}")
        
        print(f"\nAFTER:")
        print(f"  Total Area: {area_after:.2f} pixels² ({area_after/1000:.4f} mm²)")
        print(f"  Avg Intensity: {intensity_after:.1f}/255")
        print(f"  Needles with material: {needles_count_after}")
        
        print(f"\nDISSOLUTION:")
        print(f"  ⭐ Area Lost: {area_lost:.2f} pixels² ({area_lost/1000:.4f} mm²)")
        print(f"  ⭐ DISSOLUTION PERCENTAGE: {dissolution_percent:.2f}%")
        print(f"  Intensity Drop: {intensity_drop:.1f} (material loss)")
        print("="*60)
        
        # Visualize - handle both triangle and non-triangle modes
        if triangles_before is not None:
            output_before = self.visualize_triangles(image_before, triangles_before, "BEFORE")
        else:
            output_before = self.visualize_results(image_before, needles_before, "BEFORE")
        
        if triangles_after is not None:
            output_after = self.visualize_triangles(image_after, triangles_after, "AFTER")
        else:
            output_after = self.visualize_results(image_after, needles_after, "AFTER")
        
        # Show comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        before_rgb = cv2.cvtColor(output_before, cv2.COLOR_BGR2RGB)
        after_rgb = cv2.cvtColor(output_after, cv2.COLOR_BGR2RGB)
        
        axes[0].imshow(before_rgb)
        axes[0].set_title(f"BEFORE: {area_before:.0f} px² ({needles_count_before} needles)")
        axes[0].axis('off')
        
        axes[1].imshow(after_rgb)
        axes[1].set_title(f"AFTER: {area_after:.0f} px² | Dissolution: {dissolution_percent:.1f}%")
        axes[1].axis('off')
        
        plt.suptitle(f"Dissolution Analysis: {dissolution_percent:.1f}% Material Loss", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


class AnalysisMenu:
    """Main menu for dissolution analysis."""
    
    def __init__(self):
        self.detector = MicroneedleDetector()
        self.root = None
    
    def show_copyright(self):
        """Show copyright message."""
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Copyright", 
                           "Triangular Microneedle Dissolution Analyzer v4\n\n"
                           "App by Dr. ali mohammad halvani\n"
                           "email: Halvaniamwork@gmail.com")
        root.destroy()
    
    def create_main_menu(self):
        """Create main menu window."""
        self.root = tk.Tk()
        self.root.title("Microneedle Dissolution Analyzer")
        self.root.geometry("500x400")
        self.root.configure(bg='#2b2b2b')
        
        # Title
        title_label = tk.Label(self.root, 
                              text="Microneedle Dissolution Analyzer",
                              font=("Arial", 16, "bold"),
                              bg='#2b2b2b',
                              fg='white')
        title_label.pack(pady=30)
        
        subtitle_label = tk.Label(self.root,
                                 text="Optimized for Triangular/Conical Needles",
                                 font=("Arial", 10),
                                 bg='#2b2b2b',
                                 fg='#cccccc')
        subtitle_label.pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(self.root, bg='#2b2b2b')
        button_frame.pack(pady=50)
        
        btn_new = tk.Button(button_frame,
                           text="New Analysis",
                           font=("Arial", 14, "bold"),
                           width=20,
                           height=2,
                           bg='#4CAF50',
                           fg='white',
                           command=self.new_analysis)
        btn_new.pack(pady=10)
        
        btn_previous = tk.Button(button_frame,
                                text="Previous Analysis",
                                font=("Arial", 14, "bold"),
                                width=20,
                                height=2,
                                bg='#2196F3',
                                fg='white',
                                command=self.previous_analysis)
        btn_previous.pack(pady=10)
        
        btn_exit = tk.Button(button_frame,
                            text="Exit",
                            font=("Arial", 14, "bold"),
                            width=20,
                            height=2,
                            bg='#f44336',
                            fg='white',
                            command=self.root.quit)
        btn_exit.pack(pady=10)
        
        # Footer
        footer_label = tk.Label(self.root,
                               text="Dr. ali mohammad halvani",
                               font=("Arial", 9),
                               bg='#2b2b2b',
                               fg='#888888')
        footer_label.pack(side=tk.BOTTOM, pady=10)
        
        self.root.mainloop()
    
    def new_analysis(self):
        """Start new analysis workflow."""
        # Ask for analysis name
        analysis_name = simpledialog.askstring("New Analysis",
                                              "Enter analysis name:",
                                              parent=self.root)
        
        if not analysis_name:
            return
        
        # Check if name already exists
        existing_analyses = self.detector.list_analyses()
        if any(a['name'] == analysis_name for a in existing_analyses):
            overwrite = messagebox.askyesno("Name Exists",
                                           f"Analysis '{analysis_name}' already exists.\n"
                                           "Do you want to overwrite it?")
            if not overwrite:
                return
        
        # Select BEFORE image
        before_image = filedialog.askopenfilename(
            title="Select BEFORE Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
            parent=self.root
        )
        
        if not before_image:
            return
        
        # Select AFTER image
        after_image = filedialog.askopenfilename(
            title="Select AFTER Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
            parent=self.root
        )
        
        if not after_image:
            return
        
        # Process analysis
        try:
            self.root.withdraw()  # Hide menu during processing
            result = self.detector.process_new_analysis(analysis_name, before_image, after_image)
            self.root.deiconify()  # Show menu again
            
            if result:
                messagebox.showinfo("Success",
                                   f"Analysis '{analysis_name}' completed successfully!",
                                   parent=self.root)
        except Exception as e:
            self.root.deiconify()
            messagebox.showerror("Error", f"Error processing analysis:\n{str(e)}",
                               parent=self.root)
            import traceback
            traceback.print_exc()
    
    def previous_analysis(self):
        """Load and re-edit previous analysis."""
        analyses = self.detector.list_analyses()
        
        if not analyses:
            messagebox.showinfo("No Analyses",
                               "No previous analyses found in database.",
                               parent=self.root)
            return
        
        # Create selection window
        select_window = tk.Toplevel(self.root)
        select_window.title("Select Previous Analysis")
        select_window.geometry("600x400")
        select_window.configure(bg='#2b2b2b')
        
        title = tk.Label(select_window,
                        text="Select an analysis to re-edit:",
                        font=("Arial", 12, "bold"),
                        bg='#2b2b2b',
                        fg='white')
        title.pack(pady=10)
        
        # Create listbox with scrollbar
        list_frame = tk.Frame(select_window, bg='#2b2b2b')
        list_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        scrollbar = Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = Listbox(list_frame,
                         yscrollcommand=scrollbar.set,
                         font=("Arial", 11),
                         height=15,
                         bg='#3b3b3b',
                         fg='white',
                         selectbackground='#4CAF50')
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=listbox.yview)
        
        # Add analyses to list
        for analysis in analyses:
            display_text = f"{analysis['name']} - {analysis['date']}"
            listbox.insert(tk.END, display_text)
        
        def on_select():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection",
                                      "Please select an analysis.",
                                      parent=select_window)
                return
            
            selected_analysis = analyses[selection[0]]
            select_window.destroy()
            
            # Process selected analysis
            try:
                self.root.withdraw()
                self.detector.process_previous_analysis(selected_analysis['name'])
                self.root.deiconify()
                
                messagebox.showinfo("Success",
                                   f"Analysis '{selected_analysis['name']}' updated successfully!",
                                   parent=self.root)
            except Exception as e:
                self.root.deiconify()
                messagebox.showerror("Error",
                                   f"Error processing analysis:\n{str(e)}",
                                   parent=self.root)
                import traceback
                traceback.print_exc()
        
        # Buttons
        button_frame = tk.Frame(select_window, bg='#2b2b2b')
        button_frame.pack(pady=10)
        
        btn_select = tk.Button(button_frame,
                              text="Select",
                              font=("Arial", 12, "bold"),
                              width=15,
                              bg='#4CAF50',
                              fg='white',
                              command=on_select)
        btn_select.pack(side=tk.LEFT, padx=10)
        
        btn_cancel = tk.Button(button_frame,
                              text="Cancel",
                              font=("Arial", 12, "bold"),
                              width=15,
                              bg='#f44336',
                              fg='white',
                              command=select_window.destroy)
        btn_cancel.pack(side=tk.LEFT, padx=10)
        
        # Double-click to select
        listbox.bind('<Double-Button-1>', lambda e: on_select())
    
    def run(self):
        """Run the application."""
        self.show_copyright()
        self.create_main_menu()


def main():
    """Main entry point."""
    app = AnalysisMenu()
    app.run()


if __name__ == "__main__":
    main()
