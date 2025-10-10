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
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    # Show password input if not authenticated
    if not st.session_state["password_correct"]:
        st.markdown("---")
        st.markdown("### 🔒 Secure Access Required")
        st.text_input(
            "Enter application password:", 
            type="password", 
            on_change=password_entered, 
            key="password",
            help="Contact administrator if you don't have the password"
        )
        
        st.markdown("---")
        return False
    
    return True

# --- Wavelength Calculation Functions ---
def calculate_dominant_wavelength(x, y, reference_white=(0.3333, 0.3333)):
    """
    Calculate dominant wavelength from CIE x,y coordinates
    """
    # Detailed spectral locus for accurate wavelength calculation
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
    
    # Simple approximation: find closest point on spectral locus
    distances = []
    for wl, (x_l, y_l) in spectral_locus.items():
        distance = np.sqrt((x - x_l)**2 + (y - y_l)**2)
        distances.append((wl, distance, x_l, y_l))
    
    # Find the closest wavelength
    closest_wl, min_distance, closest_x, closest_y = min(distances, key=lambda x: x[1])
    
    # Check if it's a purple color (inside the triangle)
    # For purple colors, the line from white point won't intersect spectral locus
    line_slope = (y - reference_white[1]) / (x - reference_white[0]) if (x - reference_white[0]) != 0 else float('inf')
    
    # Simple purple detection: if the point is to the right of the red region
    if x > 0.5 and y < 0.3:
        return "Purple (Non-spectral)", True
    
    return closest_wl, False

def calculate_color_purity(x, y, reference_white=(0.3333, 0.3333)):
    """
    Calculate color purity from CIE x,y coordinates
    """
    dominant_wl, is_complementary = calculate_dominant_wavelength(x, y, reference_white)
    
    if is_complementary or dominant_wl == "Purple (Non-spectral)":
        return 1.0  # Maximum purity for purple colors
    
    # Get spectral locus for the dominant wavelength
    spectral_locus = {
        380: (0.1741, 0.0050), 385: (0.1740, 0.0050), 390: (0.1738, 0.0049),
        # ... include all points from above
        780: (0.7347, 0.2653)
    }
    
    if dominant_wl in spectral_locus:
        x_locus, y_locus = spectral_locus[dominant_wl]
    else:
        # Find nearest wavelength
        closest_wl = min(spectral_locus.keys(), key=lambda wl: abs(wl - dominant_wl))
        x_locus, y_locus = spectral_locus[closest_wl]
    
    # Calculate purity
    distance_total = np.sqrt((x_locus - reference_white[0])**2 + (y_locus - reference_white[1])**2)
    distance_sample = np.sqrt((x - reference_white[0])**2 + (y - reference_white[1])**2)
    
    purity = distance_sample / distance_total if distance_total > 0 else 0
    return min(purity, 1.0)

# --- MAIN APPLICATION ---
if check_password():
    # Configure page - using centered layout instead of wide
    st.set_page_config(page_title="CIE Chromaticity Comparator", layout="centered")

    st.title("🔄 CIE 1931 Chromaticity Diagram Comparator")
    st.success("✅ Authentication successful! Welcome to the CIE Comparator.")

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
                                             help="Calculate dominant wavelength for all corner points")

    # Colors for different LED sets
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    border_colors = ['darkblue', 'darkred', 'darkgreen', 'darkorange', 'purple', 'saddlebrown']
    led_names = ['LED Set A', 'LED Set B', 'LED Set C', 'LED Set D', 'LED Set E', 'LED Set F']

    # --- Data input sections ---
    st.sidebar.header("📊 LED Specifications")

    all_polygons = []
    all_labels = []
    all_wavelengths = []  # Store wavelengths for all points of all polygons

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
        point_wavelengths = []  # Store wavelengths for this polygon's points
        
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
            
            # Calculate wavelength for this specific point
            if calculate_wavelength:
                dominant_wl, is_complementary = calculate_dominant_wavelength(x, y)
                purity = calculate_color_purity(x, y)
                point_wavelengths.append({
                    'point': i+1,
                    'x': x,
                    'y': y,
                    'wavelength': dominant_wl,
                    'is_complementary': is_complementary,
                    'purity': purity
                })
        
        all_polygons.append(np.array(points))
        all_labels.append(custom_name)
        all_wavelengths.append(point_wavelengths)
        
        st.sidebar.markdown("---")

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Background
    ax.imshow(rgb_bg, extent=(0, 0.8, 0, 0.9), origin='lower', aspect='auto')

    # Plot each polygon
    centroids = []
    
    for idx, (polygon, label, color, border_color, point_wavelengths) in enumerate(zip(all_polygons, all_labels, colors, border_colors, all_wavelengths)):
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
            
            # Points with wavelength annotations
            if show_points:
                for i, (point, wavelength_info) in enumerate(zip(polygon, point_wavelengths)):
                    ax.scatter(point[0], point[1], color=color, s=60, 
                            edgecolors='white', zorder=5)
                    
                    # Add wavelength annotation near each point
                    if calculate_wavelength and wavelength_info:
                        wl_text = f"P{i+1}"
                        if wavelength_info['wavelength'] != "Purple (Non-spectral)":
                            wl_text += f"\n{wavelength_info['wavelength']:.0f}nm"
                        else:
                            wl_text += f"\nPurple"
                        
                        # Offset the text slightly from the point
                        ax.annotate(wl_text, (point[0], point[1]), 
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8, color=color, weight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
            # Centroid
            if show_centroids:
                centroid = np.mean(polygon, axis=0)
                centroids.append(centroid)
                
                # Calculate wavelength for centroid
                centroid_wavelength_info = None
                if calculate_wavelength:
                    dominant_wl, is_complementary = calculate_dominant_wavelength(centroid[0], centroid[1])
                    purity = calculate_color_purity(centroid[0], centroid[1])
                    centroid_wavelength_info = {
                        'wavelength': dominant_wl,
                        'is_complementary': is_complementary,
                        'purity': purity
                    }
                
                # Add wavelength to centroid label
                centroid_label = f'{label} Centroid'
                if centroid_wavelength_info and centroid_wavelength_info['wavelength'] != "Purple (Non-spectral)":
                    centroid_label += f'\n({centroid_wavelength_info["wavelength"]:.1f} nm)'
                elif centroid_wavelength_info:
                    centroid_label += f'\n({centroid_wavelength_info["wavelength"]})'
                
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

    for idx, (tab, polygon, label, point_wavelengths) in enumerate(zip(tabs, all_polygons, all_labels, all_wavelengths)):
        with tab:
            if num_led_sets > 1:
                st.subheader(f"{label}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Coordinates & Wavelengths**")
                # Create enhanced data table with wavelengths
                data = {
                    "Point": [f"P{i+1}" for i in range(len(polygon))],
                    "x": polygon[:, 0],
                    "y": polygon[:, 1]
                }
                
                if calculate_wavelength:
                    wavelengths = []
                    for wl_info in point_wavelengths:
                        if wl_info['wavelength'] == "Purple (Non-spectral)":
                            wavelengths.append("Purple")
                        else:
                            wavelengths.append(f"{wl_info['wavelength']:.1f} nm")
                    data["Wavelength"] = wavelengths
                    
                    purities = [f"{wl_info['purity']:.3f}" for wl_info in point_wavelengths]
                    data["Purity"] = purities
                
                st.dataframe(data, use_container_width=True, hide_index=True)
            
            with col2:
                st.write("**Statistics**")
                centroid = np.mean(polygon, axis=0)
                area = 0.5 * np.abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) 
                                - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))
                
                # Calculate wavelength for centroid
                centroid_wavelength_info = None
                if calculate_wavelength:
                    dominant_wl, is_complementary = calculate_dominant_wavelength(centroid[0], centroid[1])
                    purity = calculate_color_purity(centroid[0], centroid[1])
                    centroid_wavelength_info = {
                        'wavelength': dominant_wl,
                        'is_complementary': is_complementary,
                        'purity': purity
                    }
                
                st.metric("Centroid", f"({centroid[0]:.4f}, {centroid[1]:.4f})")
                st.metric("Number of Points", len(polygon))
                st.metric("Polygon Area", f"{area:.6f}")
                
                # Display centroid wavelength information
                if centroid_wavelength_info:
                    if centroid_wavelength_info['is_complementary']:
                        st.metric("Centroid Wavelength", "Purple (Non-spectral)")
                    else:
                        st.metric("Centroid Wavelength", f"{centroid_wavelength_info['wavelength']:.1f} nm")
                    st.metric("Centroid Purity", f"{centroid_wavelength_info['purity']:.3f}")

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

    # --- Wavelength Range Analysis ---
    if calculate_wavelength and num_led_sets >= 1:
        st.header("🌈 Wavelength Range Analysis")
        
        for idx, (label, point_wavelengths) in enumerate(zip(all_labels, all_wavelengths)):
            # Extract numerical wavelengths (excluding purple)
            numerical_wls = []
            for wl_info in point_wavelengths:
                if wl_info['wavelength'] != "Purple (Non-spectral)":
                    numerical_wls.append(wl_info['wavelength'])
            
            if numerical_wls:
                min_wl = min(numerical_wls)
                max_wl = max(numerical_wls)
                wavelength_range = max_wl - min_wl
                
                st.subheader(f"{label} Wavelength Range")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min Wavelength", f"{min_wl:.1f} nm")
                with col2:
                    st.metric("Max Wavelength", f"{max_wl:.1f} nm")
                with col3:
                    st.metric("Wavelength Range", f"{wavelength_range:.1f} nm")

    # --- Quick tips ---
    with st.expander("💡 Tips for better visualization"):
        st.markdown("""
        - **Wavelength Display**: Each point now shows its wavelength in nanometers
        - **Point Labels**: P1, P2, etc. with corresponding wavelengths
        - **Purple Colors**: Displayed as "Purple" for non-spectral colors
        - **Clear View**: For crowded displays, toggle points on/off as needed
        - **Wavelength Range**: See the complete wavelength span for each LED set
        """)

    # --- Wavelength explanation ---
    with st.expander("🔬 About Wavelength Calculation"):
        st.markdown("""
        ### Point-by-Point Wavelength Analysis
        
        **Now Calculating**: Wavelengths for **ALL corner points** of each polygon
        
        **What You See**:
        - **Each point labeled** (P1, P2, P3, P4, etc.)
        - **Individual wavelengths** for every coordinate
        - **Wavelength range** for complete color gamut analysis
        - **Centroid wavelength** for average color
        
        **Technical Details**:
        - Uses CIE 1931 standard observer
        - D65 white point reference
        - Spectral locus interpolation for accuracy
        - Purple colors identified as non-spectral
        
        **Industry Application**:
        - **LED Binning**: Verify wavelength consistency across corners
        - **Color Gamut**: Understand wavelength distribution
        - **Quality Control**: Ensure all points meet specifications
        """)
