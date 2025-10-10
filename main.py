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
    
    # Initialize session state - FIXED: Use st.session_state, not st.sidebar
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
        
        # Help information
        with st.expander("🔧 Setup Information"):
            st.markdown("""
            **For Administrators:**
            - Create `.streamlit/secrets.toml` with: `password = "your_password"`
            - Or set environment variable: `CIE_APP_PASSWORD`
            - Current setup: Using default password
            """)
        
        st.markdown("---")
        return False
    
    return True

# --- Wavelength Calculation Functions ---
def calculate_dominant_wavelength(x, y, reference_white=(0.3333, 0.3333)):
    """
    Calculate dominant wavelength from CIE x,y coordinates
    """
    # Simplified spectral locus for demonstration
    spectral_locus = {
        380: (0.1741, 0.0050), 400: (0.1733, 0.0048), 420: (0.1714, 0.0051),
        440: (0.1644, 0.0109), 460: (0.1440, 0.0297), 480: (0.0913, 0.1327),
        500: (0.0082, 0.5384), 520: (0.0743, 0.8338), 540: (0.2296, 0.7543),
        560: (0.3731, 0.6245), 580: (0.5125, 0.4866), 600: (0.6270, 0.3725),
        620: (0.6915, 0.3083), 640: (0.7190, 0.2809), 660: (0.7300, 0.2700),
        680: (0.7334, 0.2666), 700: (0.7347, 0.2653)
    }
    
    # Convert to numpy arrays for calculation
    wavelengths = np.array(list(spectral_locus.keys()))
    x_locus = np.array([point[0] for point in spectral_locus.values()])
    y_locus = np.array([point[1] for point in spectral_locus.values()])
    
    # Simple approximation: find closest point on spectral locus
    distances = []
    for wl, (x_l, y_l) in spectral_locus.items():
        distance = np.sqrt((x - x_l)**2 + (y - y_l)**2)
        distances.append((wl, distance))
    
    # Find the closest wavelength
    closest_wl, min_distance = min(distances, key=lambda x: x[1])
    
    # Check if it's a purple color (inside the triangle)
    # Simple check: if y coordinate is low and x is moderate to high
    if y < 0.3 and x > 0.3:
        return "Purple (Non-spectral)", True
    
    return closest_wl, False

def calculate_color_purity(x, y, reference_white=(0.3333, 0.3333)):
    """
    Calculate color purity from CIE x,y coordinates
    """
    dominant_wl, is_complementary = calculate_dominant_wavelength(x, y, reference_white)
    
    if is_complementary or dominant_wl == "Purple (Non-spectral)":
        return 1.0  # Maximum purity for purple colors
    
    # Simplified purity calculation
    spectral_locus = {
        380: (0.1741, 0.0050), 400: (0.1733, 0.0048), 420: (0.1714, 0.0051),
        440: (0.1644, 0.0109), 460: (0.1440, 0.0297), 480: (0.0913, 0.1327),
        500: (0.0082, 0.5384), 520: (0.0743, 0.8338), 540: (0.2296, 0.7543),
        560: (0.3731, 0.6245), 580: (0.5125, 0.4866), 600: (0.6270, 0.3725),
        620: (0.6915, 0.3083), 640: (0.7190, 0.2809), 660: (0.7300, 0.2700),
        680: (0.7334, 0.2666), 700: (0.7347, 0.2653)
    }
    
    # Find closest wavelength in spectral locus
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
                elif wavelength_info:
                    centroid_label += f'\n({wavelength_info["wavelength"]})'
                
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
        1. Find closest point on spectral locus (rainbow edge)
        2. Wavelength at that point = Dominant wavelength
        3. Distance ratio = Color purity
        
        *Note: This uses simplified calculation for demonstration. Professional applications may require more precise methods.*
        """)
