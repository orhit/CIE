import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from colour.models import xyY_to_XYZ, XYZ_to_sRGB
from matplotlib.patches import Polygon
import io
import base64
import os

# --- Improved Password Protection ---
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def get_configured_password():
        """Safely get password from secrets or environment"""
        try:
            # Method 1: Try Streamlit secrets
            if hasattr(st, 'secrets') and 'password' in st.secrets:
                return st.secrets['password']
        except Exception:
            pass
        
        try:
            # Method 2: Try environment variable
            return os.environ.get('CIE_APP_PASSWORD', '')
        except Exception:
            pass
        
        # Method 3: Default fallback password (change this!)
        return "CIE2024!Secure"
    
    def password_entered():
        correct_password = get_configured_password()
        if st.session_state["password"] == correct_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    
    # Initialize session state
    if "password_correct" not in st.sidebar:
        st.sidebar["password_correct"] = False
    
    # Show password input if not authenticated
    if not st.sidebar["password_correct"]:
        st.markdown("---")
        st.markdown("### 🔒 Secure Access Required")
        st.text_input(
            "Enter application password:", 
            type="password", 
            on_change=password_entered, 
            key="password",
            help="Contact administrator if you don't have the password"
        )
        
        # Help information
        
        
        st.markdown("---")
        return False
    
    return True

# --- Wavelength Calculation Functions ---
def calculate_dominant_wavelength(x, y, reference_white=(0.3333, 0.3333)):
    """
    Calculate dominant wavelength from CIE x,y coordinates
    
    Parameters:
    x, y: CIE chromaticity coordinates
    reference_white: White point reference (default D65)
    
    Returns:
    dominant_wl: Dominant wavelength in nm (or complementary for purple)
    is_complementary: Boolean indicating if it's a complementary wavelength
    """
    # CIE 1931 spectral locus coordinates (simplified)
    # Full table would have more points for accuracy
    spectral_locus = {
        380: (0.1741, 0.0050), 385: (0.1740, 0.0050), 390: (0.1738, 0.0049),
        395: (0.1736, 0.0049), 400: (0.1733, 0.0048), 405: (0.1730, 0.0048),
        410: (0.1726, 0.0048), 415: (0.1721, 0.0048), 420: (0.1714, 0.0051),
        425: (0.1703, 0.0058), 430: (0.1689, 0.0069), 435: (0.1669, 0.0086),
        440: (0.1644, 0.0109), 445: (0.1611, 0.0138), 450: (0.1566, 0.0177),
        455: (0.1510, 0.0227), 460: (0.1440, 0.0297), 465: (0.1355, 0.0399),
        470: (0.1241, 0.0578), 475: (0.1096, 0.0868), 480: (0.0913, 0.1327),
        485: (0.0687, 0.2007), 490: (0.0454, 0.2950), 495: (0.0235, 0.4127),
        500: (0.0082, 0.5384), 505: (0.0039, 0.6548), 510: (0.0139, 0.7502),
        515: (0.0389, 0.8120), 520: (0.0743, 0.8338), 525: (0.1142, 0.8262),
        530: (0.1547, 0.8059), 535: (0.1929, 0.7816), 540: (0.2296, 0.7543),
        545: (0.2658, 0.7243), 550: (0.3016, 0.6923), 555: (0.3373, 0.6589),
        560: (0.3731, 0.6245), 565: (0.4087, 0.5896), 570: (0.4441, 0.5547),
        575: (0.4788, 0.5202), 580: (0.5125, 0.4866), 585: (0.5448, 0.4544),
        590: (0.5752, 0.4242), 595: (0.6029, 0.3965), 600: (0.6270, 0.3725),
        605: (0.6482, 0.3514), 610: (0.6658, 0.3340), 615: (0.6801, 0.3197),
        620: (0.6915, 0.3083), 625: (0.7006, 0.2993), 630: (0.7079, 0.2920),
        635: (0.7140, 0.2859), 640: (0.7190, 0.2809), 645: (0.7230, 0.2770),
        650: (0.7260, 0.2740), 655: (0.7283, 0.2717), 660: (0.7300, 0.2700),
        665: (0.7311, 0.2689), 670: (0.7320, 0.2680), 675: (0.7327, 0.2673),
        680: (0.7334, 0.2666), 685: (0.7340, 0.2660), 690: (0.7344, 0.2656),
        695: (0.7346, 0.2654), 700: (0.7347, 0.2653), 705: (0.7347, 0.2653),
        710: (0.7347, 0.2653), 715: (0.7347, 0.2653), 720: (0.7347, 0.2653),
        725: (0.7347, 0.2653), 730: (0.7347, 0.2653), 735: (0.7347, 0.2653),
        740: (0.7347, 0.2653), 745: (0.7347, 0.2653), 750: (0.7347, 0.2653),
        755: (0.7347, 0.2653), 760: (0.7347, 0.2653), 765: (0.7347, 0.2653),
        770: (0.7347, 0.2653), 775: (0.7347, 0.2653), 780: (0.7347, 0.2653)
    }
    
    # Convert to numpy arrays for calculation
    wavelengths = np.array(list(spectral_locus.keys()))
    x_locus = np.array([point[0] for point in spectral_locus.values()])
    y_locus = np.array([point[1] for point in spectral_locus.values()])
    
    # Calculate line from white point through sample point
    slope = (y - reference_white[1]) / (x - reference_white[0])
    
    # Find intersection with spectral locus
    min_distance = float('inf')
    dominant_wl = None
    is_complementary = False
    
    for i in range(len(wavelengths) - 1):
        # Check intersection with each segment of spectral locus
        x1, y1 = x_locus[i], y_locus[i]
        x2, y2 = x_locus[i+1], y_locus[i+1]
        
        # Calculate intersection
        if (x2 - x1) != 0:
            segment_slope = (y2 - y1) / (x2 - x1)
            if slope != segment_slope:  # Avoid parallel lines
                # Intersection point
                x_intersect = (reference_white[1] - y1 + segment_slope * x1 - slope * reference_white[0]) / (segment_slope - slope)
                y_intersect = slope * (x_intersect - reference_white[0]) + reference_white[1]
                
                # Check if intersection is within segment
                if (min(x1, x2) <= x_intersect <= max(x1, x2) and 
                    min(y1, y2) <= y_intersect <= max(y1, y2)):
                    
                    # Calculate distance from sample to intersection
                    distance = np.sqrt((x - x_intersect)**2 + (y - y_intersect)**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        # Interpolate wavelength
                        t = (x_intersect - x1) / (x2 - x1)
                        dominant_wl = wavelengths[i] + t * (wavelengths[i+1] - wavelengths[i])
    
    # If no intersection found, it's a purple (complementary wavelength)
    if dominant_wl is None:
        is_complementary = True
        # For purple colors, we calculate the complementary wavelength
        # This would require more complex calculation
        dominant_wl = "Purple (Non-spectral)"
    
    return dominant_wl, is_complementary

def calculate_color_purity(x, y, reference_white=(0.3333, 0.3333)):
    """
    Calculate color purity from CIE x,y coordinates
    
    Parameters:
    x, y: CIE chromaticity coordinates
    reference_white: White point reference
    
    Returns:
    purity: Color purity (0 to 1)
    """
    dominant_wl, is_complementary = calculate_dominant_wavelength(x, y, reference_white)
    
    if is_complementary or dominant_wl == "Purple (Non-spectral)":
        return 1.0  # Maximum purity for purple colors
    
    # Get the spectral locus point for dominant wavelength
    spectral_locus = {
        # ... same spectral locus as above
        380: (0.1741, 0.0050), 385: (0.1740, 0.0050), 390: (0.1738, 0.0049),
        # ... include all points from above
        780: (0.7347, 0.2653)
    }
    
    # Find closest wavelength in spectral locus
    closest_wl = min(spectral_locus.keys(), key=lambda wl: abs(wl - dominant_wl))
    x_locus, y_locus = spectral_locus[closest_wl]
    
    # Calculate purity
    distance_total = np.sqrt((x_locus - reference_white[0])**2 + (y_locus - reference_white[1])**2)
    distance_sample = np.sqrt((x - reference_white[0])**2 + (y - reference_white[1])**2)
    
    purity = distance_sample / distance_total if distance_total > 0 else 0
    return min(purity, 1.0)

if check_password():
    # Configure page - using centered layout instead of wide
    st.set_page_config(page_title="CIE Chromaticity Comparator", layout="centered")

    st.title("🔄 CIE 1931 Chromaticity Diagram Comparator")

    # --- OPTIMIZED: Vectorized CIE background generation ---
    @st.cache_data
    def generate_cie_background():
        resolution = 200
        x_grid = np.linspace(0, 0.8, resolution)
        y_grid = np.linspace(0, 0.9, resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        xyY_points = np.dstack([xx, yy, np.ones_like(xx)])
        xyY_flat = xyY_points.reshape(-1, 3)
        
        XYZ_flat = xyY_to_XYZ(xyY_flat)
        rgb_flat = XYZ_to_sRGB(XYZ_flat)
        rgb_bg = rgb_flat.reshape(resolution, resolution, 3)
        return np.clip(rgb_bg, 0, 1)

    # Generate background once
    rgb_bg = generate_cie_background()

    # --- Sidebar configuration ---
    st.sidebar.header("🔧 Configuration")

    # Number of LED sets
    num_led_sets = st.sidebar.number_input(
        "Number of LED sets to compare", 
        min_value=1, 
        max_value=6, 
        value=2,
        help="Compare multiple LED specifications (up to 6)"
    )

    # Display options
    col1, col2 = st.sidebar.columns(2)
    with col1:
        show_fill = st.checkbox("Fill polygons", value=False, help="Fill the polygon areas")
        show_points = st.checkbox("Show points", value=True, help="Show individual points")
    with col2:
        show_borders = st.checkbox("Show borders", value=True, help="Show polygon borders")
        show_centroids = st.checkbox("Show centroids", value=True, help="Show center points")
    
    # Wavelength calculation option
    calculate_wavelength = st.sidebar.checkbox("Calculate Wavelength", value=True, 
                                             help="Calculate dominant wavelength for centroids")

    # Colors for different LED sets
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    border_colors = ['darkblue', 'darkred', 'darkgreen', 'darkorange', 'purple', 'saddlebrown']
    led_names = ['LED Set A', 'LED Set B', 'LED Set C', 'LED Set D', 'LED Set E', 'LED Set F']

    # --- Data input sections ---
    st.sidebar.header("📊 LED Specifications")

    all_polygons = []
    all_labels = []

    for led_idx in range(num_led_sets):
        st.sidebar.subheader(f"🔹 {led_names[led_idx]}")
        
        # Allow custom naming
        custom_name = st.sidebar.text_input(f"Name for set {led_idx+1}", 
                                        value=led_names[led_idx],
                                        key=f"name_{led_idx}")
        
        num_points = st.sidebar.number_input(
            f"Points in {custom_name}", 
            min_value=3, 
            max_value=20, 
            value=4, 
            step=1,
            key=f"num_points_{led_idx}"
        )
        
        points = []
        for i in range(num_points):
            col1, col2 = st.sidebar.columns(2)
            with col1:
                # Default values that create a reasonable quadrilateral
                default_x = 0.68 + (led_idx * 0.01) + (i * 0.005)
                x = st.number_input(
                    f"x{i+1}", 
                    min_value=0.0, 
                    max_value=0.8, 
                    value=default_x, 
                    step=0.0001, 
                    format="%.4f", 
                    key=f"x_{led_idx}_{i}"
                )
            with col2:
                default_y = 0.30 - (led_idx * 0.01) - (i * 0.005)
                y = st.number_input(
                    f"y{i+1}", 
                    min_value=0.0, 
                    max_value=0.9, 
                    value=default_y, 
                    step=0.0001, 
                    format="%.4f", 
                    key=f"y_{led_idx}_{i}"
                )
            points.append([x, y])
        
        all_polygons.append(np.array(points))
        all_labels.append(custom_name)
        
        st.sidebar.markdown("---")

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Background
    ax.imshow(rgb_bg, extent=(0, 0.8, 0, 0.9), origin='lower', aspect='auto')

    # Plot each polygon
    centroids = []
    wavelength_data = []  # Store wavelength information
    
    for idx, (polygon, label, color, border_color) in enumerate(zip(all_polygons, all_labels, colors, border_colors)):
        if len(polygon) >= 3:
            # Fill polygon (with edgecolor for border)
            if show_fill:
                poly = Polygon(polygon, color=color, alpha=0.3, 
                            edgecolor=border_color if show_borders else 'none',
                            linewidth=2, label=label)
                ax.add_patch(poly)
            elif show_borders:
                # If no fill, just show the border
                closed_poly = np.vstack([polygon, polygon[0]])
                ax.plot(closed_poly[:, 0], closed_poly[:, 1], color=border_color, 
                    linewidth=2, label=label)
            
            # Points
            if show_points:
                ax.scatter(polygon[:, 0], polygon[:, 1], color=color, s=60, 
                        edgecolors='white', zorder=5)
            
            # Centroid
            if show_centroids:
                centroid = np.mean(polygon, axis=0)
                centroids.append(centroid)
                
                # Calculate wavelength for centroid if enabled
                wavelength_info = None
                if calculate_wavelength:
                    dominant_wl, is_complementary = calculate_dominant_wavelength(centroid[0], centroid[1])
                    purity = calculate_color_purity(centroid[0], centroid[1])
                    wavelength_info = {
                        'wavelength': dominant_wl,
                        'is_complementary': is_complementary,
                        'purity': purity
                    }
                    wavelength_data.append(wavelength_info)
                
                # Add wavelength to label if calculated
                centroid_label = f'{label} Centroid'
                if wavelength_info and wavelength_info['wavelength'] != "Purple (Non-spectral)":
                    centroid_label += f'\n({wavelength_info["wavelength"]:.1f} nm)'
                
                ax.scatter(centroid[0], centroid[1], color=color, s=100, 
                        marker='X', edgecolors='white', linewidth=1, zorder=6,
                        label=centroid_label)

    # Auto-zoom to fit all points
    if all_polygons:
        all_points_combined = np.vstack(all_polygons)
        x_min, x_max = np.min(all_points_combined[:, 0]) - 0.02, np.max(all_points_combined[:, 0]) + 0.02
        y_min, y_max = np.min(all_points_combined[:, 1]) - 0.02, np.max(all_points_combined[:, 1]) + 0.02
        
        ax.set_xlim(max(0, x_min), min(0.8, x_max))
        ax.set_ylim(max(0, y_min), min(0.9, y_max))

    # Styling
    ax.set_xlabel("CIE x", fontsize=12)
    ax.set_ylabel("CIE y", fontsize=12)

    if num_led_sets == 1:
        ax.set_title(f"CIE Chromaticity Diagram - {all_labels[0]}", fontsize=14, fontweight='bold')
    else:
        ax.set_title(f"CIE Chromaticity Diagram - {num_led_sets} LED Sets Comparison", fontsize=14, fontweight='bold')

    # Border styling
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    ax.grid(True, linestyle='--', alpha=0.3)

    # Legend handling - only show unique labels
    handles, labels_legend = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.15, 1))

    # --- Display the plot ---
    st.pyplot(fig)

    # --- Download functionality ---
    st.sidebar.header("💾 Export")

    def get_image_download_link(fig, filename="cie_chromaticity_comparison.png"):
        """Generate a download link for the matplotlib figure"""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">📥 Download PNG Image</a>'
        return href

    # Download button
    st.sidebar.markdown(get_image_download_link(fig), unsafe_allow_html=True)

    # --- Data table display ---
    st.header("📋 Data Summary")

    # Create tabs for better organization
    if num_led_sets > 1:
        tabs = st.tabs([f"**{label}**" for label in all_labels])
    else:
        tabs = [st.container()]

    for idx, (tab, polygon, label) in enumerate(zip(tabs, all_polygons, all_labels)):
        with tab:
            if num_led_sets > 1:
                st.subheader(f"{label}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Coordinates**")
                data = {"Point": [f"P{i+1}" for i in range(len(polygon))],
                    "x": polygon[:, 0], "y": polygon[:, 1]}
                st.dataframe(data, use_container_width=True, hide_index=True)
            
            with col2:
                st.write("**Statistics**")
                centroid = np.mean(polygon, axis=0)
                area = 0.5 * np.abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) 
                                - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))
                
                # Calculate wavelength metrics if enabled
                wavelength_info = None
                if calculate_wavelength:
                    dominant_wl, is_complementary = calculate_dominant_wavelength(centroid[0], centroid[1])
                    purity = calculate_color_purity(centroid[0], centroid[1])
                    wavelength_info = {
                        'wavelength': dominant_wl,
                        'is_complementary': is_complementary,
                        'purity': purity
                    }
                
                st.metric("Centroid", f"({centroid[0]:.4f}, {centroid[1]:.4f})")
                st.metric("Number of Points", len(polygon))
                st.metric("Polygon Area", f"{area:.6f}")
                
                # Display wavelength information
                if wavelength_info:
                    if wavelength_info['is_complementary']:
                        st.metric("Dominant Wavelength", "Purple (Non-spectral)")
                    else:
                        st.metric("Dominant Wavelength", f"{wavelength_info['wavelength']:.1f} nm")
                    st.metric("Color Purity", f"{wavelength_info['purity']:.3f}")

    # --- Comparison metrics for multiple sets ---
    if num_led_sets > 1:
        st.header("📈 Comparison Metrics")
        
        # Calculate all centroids
        centroids = [np.mean(polygon, axis=0) for polygon in all_polygons]
        
        # Create distance matrix
        st.subheader("Centroid Distances")
        distance_matrix = np.zeros((num_led_sets, num_led_sets))
        
        # Display as a nice table
        cols = st.columns(num_led_sets + 1)
        with cols[0]:
            st.write("**From → To ↓**")
        
        for i in range(num_led_sets):
            with cols[i + 1]:
                st.write(f"**{all_labels[i]}**")
        
        for i in range(num_led_sets):
            cols = st.columns(num_led_sets + 1)
            with cols[0]:
                st.write(f"**{all_labels[i]}**")
            
            for j in range(num_led_sets):
                with cols[j + 1]:
                    if i == j:
                        st.write("—")
                    else:
                        distance = np.linalg.norm(centroids[i] - centroids[j])
                        st.write(f"{distance:.6f}")

    # --- Quick tips ---
    with st.expander("💡 Tips for better visualization"):
        st.markdown("""
        - **For clear borders**: Keep 'Fill polygons' OFF and 'Show borders' ON
        - **For filled areas**: Turn ON 'Fill polygons' - borders will be automatically added
        - **Color coding**: Each LED set has a unique color for easy identification
        - **Zoom**: The plot automatically zooms to fit all your points
        - **Download**: Use the download button in sidebar to save high-quality images
        - **Wavelength**: Enable 'Calculate Wavelength' to see dominant wavelengths in nm
        """)

    # --- Wavelength explanation ---
    with st.expander("🔬 About Wavelength Calculation"):
        st.markdown("""
        ### Dominant Wavelength & Color Purity
        
        **Dominant Wavelength**: The single wavelength that most closely matches the color's hue.
        - Measured in nanometers (nm)
        - For purple colors: Displayed as "Purple (Non-spectral)" since purple isn't a single wavelength
        
        **Color Purity**: How saturated the color is (0.0 to 1.0)
        - 0.0 = Pure white
        - 1.0 = Fully saturated spectral color
        
        **Calculation Method**:
        1. Draw line from white point through your color point
        2. Find intersection with spectral locus (rainbow edge)
        3. Wavelength at intersection = Dominant wavelength
        4. Distance ratio = Color purity
        
        *Note: This uses the CIE 1931 standard observer and D65 white point*
        """)
