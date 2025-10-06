import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from colour.models import xyY_to_XYZ, XYZ_to_sRGB
from matplotlib.patches import Polygon
import io
import base64
import streamlit_authenticator as stauth
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
        
        # Help information
       
        
        st.markdown("---")
        return False
    
    return True
        
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
                ax.scatter(centroid[0], centroid[1], color=color, s=100, 
                        marker='X', edgecolors='white', linewidth=1, zorder=6,
                        label=f'{label} Centroid')

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
                
                st.metric("Centroid", f"({centroid[0]:.4f}, {centroid[1]:.4f})")
                st.metric("Number of Points", len(polygon))
                st.metric("Polygon Area", f"{area:.6f}")

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
        """)
