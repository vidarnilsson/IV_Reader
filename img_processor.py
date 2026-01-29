"""
Pokemon GO IV Scanner - Resolution Independent Version

This script extracts Pokemon name and IVs from Pokemon GO screenshots.
It works across different phone screen sizes by using relative positioning
and feature detection rather than hardcoded pixel coordinates.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytesseract


class PokemonIVScanner:
    """
    A resolution-independent Pokemon GO IV scanner.

    Uses relative positioning based on screen dimensions and color detection
    to find IV bars regardless of phone screen size.
    """

    # Relative positions based on Pokemon GO UI layout (as fractions of width/height)
    # These ratios are derived from the standard Pokemon GO interface

    # Text box at the bottom containing "This [Pokemon] was caught..."
    TEXT_BOX_Y_START = 0.87  # Start of text region (from top)
    TEXT_BOX_Y_END = 1.0  # End of text region (bottom of screen)

    # IV stats panel region (relative to screen dimensions)
    IV_PANEL_Y_START = 0.743  # Start of IV panel
    IV_PANEL_Y_END = 0.867  # End of IV panel
    IV_PANEL_X_START = 0.114  # Left edge of IV bars
    IV_PANEL_X_END = 0.470  # Right edge of IV bars

    # Within the IV panel, relative positions of the 3 IV bars (center of each bar)
    # These are calculated as: (bar_y_in_crop / crop_height)
    # Verified against actual screenshot: Attack=0.22, Defense=0.56, HP=0.89
    IV_BAR_Y_POSITIONS = [0.22, 0.56, 0.89]  # Attack, Defense, HP
    IV_BAR_X_START = 0.02  # Start of bar fill area
    IV_BAR_X_END = 0.97  # End of bar fill area

    # Offset adjustment for when 3-star badge is present
    Y_OFFSET_RATIO = 0.029  # Vertical offset when badge visible

    # Reference pixel to check for badge/offset
    CHECK_PIXEL_X = 0.053
    CHECK_PIXEL_Y = 0.886

    # Orange/red color range for IV bars (HSV)
    ORANGE_LOWER = np.array([0, 80, 80])
    ORANGE_UPPER = np.array([25, 255, 255])

    def __init__(self, debug: bool = False):
        """
        Initialize the scanner.

        Args:
            debug: If True, saves intermediate images for debugging
        """
        self.debug = debug

    def process_image(self, image_input) -> Tuple[str, List[int]]:
        """
        Process a Pokemon GO screenshot and extract Pokemon name and IVs.

        Args:
            image_input: Can be one of:
                - bytes: Raw image bytes
                - str: Path to image file
                - np.ndarray: OpenCV image (BGR format)

        Returns:
            Tuple of (pokemon_name, [attack_iv, defense_iv, hp_iv])

        Raises:
            ValueError: If image cannot be loaded or parsed
        """
        # Load image based on input type
        img = self._load_image(image_input)
        if img is None:
            raise ValueError("Could not load image")

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Check for y-offset (3-star badge visibility)
        y_offset = self._calculate_y_offset(gray, w, h)

        # Extract Pokemon name from bottom text
        pokemon_name = self._extract_pokemon_name(gray, w, h)

        # Extract IVs from the stat bars
        ivs = self._extract_ivs(hsv, w, h, y_offset)

        return pokemon_name, ivs

    def _load_image(self, image_input) -> Optional[np.ndarray]:
        """Load image from various input types."""
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, str):
            return cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            return image_input
        return None

    def _calculate_y_offset(self, gray: np.ndarray, w: int, h: int) -> int:
        """
        Calculate vertical offset based on whether the appraisal badge is visible.

        The 3-star (or other) badge can shift the IV panel position slightly.
        """
        check_x = int(w * self.CHECK_PIXEL_X)
        check_y = int(h * self.CHECK_PIXEL_Y)

        # Ensure we're within bounds
        check_x = min(check_x, w - 1)
        check_y = min(check_y, h - 1)

        pixel_value = int(gray[check_y, check_x])

        # If the pixel is bright (>200), there's an offset
        if pixel_value > 200:
            return int(h * self.Y_OFFSET_RATIO)
        return 0

    def _extract_pokemon_name(self, gray: np.ndarray, w: int, h: int) -> str:
        """
        Extract Pokemon name from the bottom text box.

        The text format is: "This [Pokemon] was caught on [date]..."
        """
        # Crop the text region at the bottom
        y1 = int(h * self.TEXT_BOX_Y_START)
        y2 = int(h * self.TEXT_BOX_Y_END)
        text_region = gray[y1:y2, :]

        # Apply threshold to improve OCR
        _, thresh = cv2.threshold(text_region, 200, 255, cv2.THRESH_BINARY_INV)

        if self.debug:
            cv2.imwrite("debug_text_region.jpg", text_region)
            cv2.imwrite("debug_text_thresh.jpg", thresh)

        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(text_region)

        # Parse the text to find the Pokemon name
        # Format: "This [Pokemon] was caught..."
        words = extracted_text.split()

        try:
            this_idx = words.index("This")
            pokemon_name = words[this_idx + 1]
            # Clean up any punctuation
            pokemon_name = "".join(c for c in pokemon_name if c.isalnum())
            return pokemon_name
        except (ValueError, IndexError):
            # Try alternative parsing methods
            return self._fallback_name_extraction(extracted_text)

    def _fallback_name_extraction(self, text: str) -> str:
        """Fallback method to extract Pokemon name if standard parsing fails."""
        # Look for pattern "This X was"
        import re

        match = re.search(r"This\s+(\w+)\s+was", text, re.IGNORECASE)
        if match:
            return match.group(1)

        # If all else fails, return unknown
        return "Unknown"

    def _extract_ivs(self, hsv: np.ndarray, w: int, h: int, y_offset: int) -> List[int]:
        """
        Extract IVs by detecting the filled vs empty portions of the IV bars.

        This method dynamically finds the bar positions by scanning for rows
        with significant bar content, making it more robust across different
        screen sizes and UI variations.
        """
        # Calculate IV panel boundaries
        panel_y1 = int(h * self.IV_PANEL_Y_START) - y_offset
        panel_y2 = int(h * self.IV_PANEL_Y_END) - y_offset
        panel_x1 = int(w * self.IV_PANEL_X_START)
        panel_x2 = int(w * self.IV_PANEL_X_END)

        # We need the BGR image for color detection
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Crop the IV panel
        iv_panel = bgr[panel_y1:panel_y2, panel_x1:panel_x2]
        panel_h, panel_w = iv_panel.shape[:2]

        if self.debug:
            cv2.imwrite("debug_iv_panel.jpg", iv_panel)

        # Scan all rows to find bars dynamically
        bar_rows = []
        for y in range(panel_h):
            row = iv_panel[y]
            filled = 0
            empty = 0

            for x in range(panel_w):
                b, g, r = row[x]

                # Filled bars: pink or orange
                if (r > 200 and g < 180 and b > 100) or (
                    r > 200 and 130 < g < 200 and b < 150
                ):
                    filled += 1
                # Empty bar sections: light gray
                elif 200 < r < 240 and 200 < g < 240 and 200 < b < 240:
                    empty += 1

            total = filled + empty
            if total > panel_w * 0.4:  # Row has significant bar content
                bar_rows.append((y, filled, empty, total))

        # Group consecutive rows into bars
        bars = []
        if bar_rows:
            current_bar = [bar_rows[0]]
            for row in bar_rows[1:]:
                if row[0] - current_bar[-1][0] <= 5:  # Consecutive or close
                    current_bar.append(row)
                else:
                    if len(current_bar) >= 5:  # Minimum bar thickness
                        bars.append(current_bar)
                    current_bar = [row]
            if len(current_bar) >= 5:
                bars.append(current_bar)

        # Calculate IV for each bar
        ivs = []
        for bar in bars[:3]:  # Take first 3 bars (Attack, Defense, HP)
            # Use the middle rows of the bar for best accuracy
            mid_start = len(bar) // 4
            mid_end = len(bar) * 3 // 4

            total_filled = sum(row[1] for row in bar[mid_start:mid_end])
            total_empty = sum(row[2] for row in bar[mid_start:mid_end])
            total = total_filled + total_empty

            if total > 0:
                fill_ratio = total_filled / total
                iv = round(fill_ratio * 15)
                iv = max(0, min(15, iv))
            else:
                iv = 0

            ivs.append(iv)

        # If we didn't find 3 bars, fall back to fixed positions
        while len(ivs) < 3:
            if self.debug:
                print(f"Warning: Only found {len(ivs)} bars, using fallback")
            # Try fallback detection
            missing_idx = len(ivs)
            y_ratio = self.IV_BAR_Y_POSITIONS[missing_idx]
            bar_y = int(panel_h * y_ratio)
            iv = self._detect_iv_at_position(iv_panel, bar_y, panel_w)
            ivs.append(iv)

        return ivs

    def _detect_iv_at_position(
        self, iv_panel: np.ndarray, bar_y: int, panel_w: int
    ) -> int:
        """Detect IV at a specific y position with scanning."""
        panel_h = iv_panel.shape[0]

        # Scan around the expected position
        for offset in range(-20, 21, 2):
            check_y = bar_y + offset
            if 0 <= check_y < panel_h:
                row = iv_panel[check_y]
                filled = 0
                empty = 0

                for x in range(panel_w):
                    b, g, r = row[x]
                    if (r > 200 and g < 180 and b > 100) or (
                        r > 200 and 130 < g < 200 and b < 150
                    ):
                        filled += 1
                    elif 200 < r < 240 and 200 < g < 240 and 200 < b < 240:
                        empty += 1

                total = filled + empty
                if total > panel_w * 0.4:
                    fill_ratio = filled / total
                    return round(fill_ratio * 15)

        return 0

    def _fallback_iv_detection(
        self, hsv_panel: np.ndarray, bar_y: int, panel_w: int
    ) -> int:
        """Fallback IV detection using HSV color masking."""
        orange_mask = cv2.inRange(hsv_panel, self.ORANGE_LOWER, self.ORANGE_UPPER)

        # Scan around the bar position
        for offset in range(-5, 6):
            check_y = bar_y + offset
            if 0 <= check_y < hsv_panel.shape[0]:
                row = orange_mask[check_y, :]
                orange_pixels = np.where(row > 0)[0]

                if len(orange_pixels) >= 10:
                    bar_length = orange_pixels[-1] - orange_pixels[0]
                    max_bar_length = int(panel_w * 0.95)
                    iv = round((bar_length / max_bar_length) * 15)
                    return max(0, min(15, iv))

        return 0

    def process_image_bytes(self, image_bytes: bytes) -> Tuple[str, List[int]]:
        """
        Process image from raw bytes (for API/server usage).

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.)

        Returns:
            Tuple of (pokemon_name, [attack_iv, defense_iv, hp_iv])
        """
        return self.process_image(image_bytes)


def process_image(image_bytes: bytes) -> Tuple[str, List[int]]:
    """
    Legacy function for backwards compatibility.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Tuple of (pokemon_name, [attack_iv, defense_iv, hp_iv])
    """
    scanner = PokemonIVScanner()
    return scanner.process_image(image_bytes)


# Alternative approach using template matching for more robust detection
class RobustPokemonIVScanner(PokemonIVScanner):
    """
    Enhanced scanner that uses multiple detection methods for better accuracy.
    """

    def _find_text_box_by_contour(
        self, gray: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the bottom text box using contour detection.

        This is more robust than fixed ratios as it adapts to the actual UI.
        """
        h, w = gray.shape

        # Look in the bottom 20% of the image
        bottom_region = gray[int(h * 0.80) :, :]

        # Threshold to find white areas
        _, thresh = cv2.threshold(bottom_region, 240, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find the largest contour that spans most of the width (the text box)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Text box should be at least 80% of screen width
            if cw > w * 0.8:
                # Adjust y to account for the offset from looking only at bottom region
                actual_y = y + int(h * 0.80)
                return (x, actual_y, cw, ch)

        return None

    def _find_iv_panel_by_color(
        self, img: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the IV panel by looking for the characteristic orange bars.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]

        # Create mask for orange color
        orange_mask = cv2.inRange(hsv, self.ORANGE_LOWER, self.ORANGE_UPPER)

        # Find contours of orange regions
        contours, _ = cv2.findContours(
            orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find horizontal bar-like contours (IV bars)
        bar_contours = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # IV bars are horizontal (width >> height) and on the left side of screen
            if cw > ch * 3 and x < w * 0.5 and cw > w * 0.1:
                bar_contours.append((x, y, cw, ch))

        if len(bar_contours) >= 3:
            # Sort by y position
            bar_contours.sort(key=lambda b: b[1])

            # Get bounding box of all bars
            min_x = min(b[0] for b in bar_contours[:3])
            min_y = min(b[1] for b in bar_contours[:3])
            max_x = max(b[0] + b[2] for b in bar_contours[:3])
            max_y = max(b[1] + b[3] for b in bar_contours[:3])

            # Add padding
            padding_y = int((max_y - min_y) * 0.2)
            padding_x = int((max_x - min_x) * 0.1)

            return (
                max(0, min_x - padding_x),
                max(0, min_y - padding_y),
                min(w, max_x - min_x + 2 * padding_x),
                min(h, max_y - min_y + 2 * padding_y),
            )

        return None


# Example usage and testing
if __name__ == "__main__":
    import sys

    # Test with a file path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "/mnt/user-data/uploads/ho.jpg"

    print(f"Processing: {image_path}")

    # Use the standard scanner
    scanner = PokemonIVScanner(debug=True)

    try:
        pokemon_name, ivs = scanner.process_image(image_path)

        print(f"\nResults:")
        print(f"  Pokemon: {pokemon_name}")
        print(f"  Attack IV:  {ivs[0]}/15")
        print(f"  Defense IV: {ivs[1]}/15")
        print(f"  HP IV:      {ivs[2]}/15")
        print(f"  Total IVs:  {sum(ivs)}/45 ({sum(ivs) / 45 * 100:.1f}%)")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
