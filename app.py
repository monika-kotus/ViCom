import hmac
import streamlit as st

def check_password():
    """Returns `True` if the user has entered the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password with compact welcome message.
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 10px;">
            <h3 style="color: #000056;">🚀 Welcome to ViCom CollabMap! 🚀</h3>
            <p style="font-size: 16px; color: #333;">
                This project is aimed at visually presenting <strong>existing collaborations</strong> 
                within the ViCom research group and uncovering <strong>potential new connections</strong>.
            </p>
        </div>
        
        <h5 style="color: #000056;">Let’s be real for a second:</h5>
        <p style="font-size: 14px; line-height: 1.3em; color: #333;">
            This was a <strong>3-week project</strong>, primarily focused on <strong>data science</strong>, not web development.
            So, while I’ve done my best to make it work smoothly, there may still be the occasional bug 
            or something that looks a little off. I hope you can bear with me! 😊
        </p>
        <p style="font-size: 14px; line-height: 1.3em; color: #333;">
            Also, I’m not a linguist, and some of the terms used in this field might have flown right over my head. 
            Despite that, I’ve tried to build something that will help you better see relationships between researchers 
            and perhaps spark new ideas for collaboration.
        </p>

        <h5 style="color: #000056;">How were the expertise profiles generated?</h5>
        <p style="font-size: 14px; line-height: 1.3em; color: #333;">
            To generate the expertise profiles, I used a <strong>large language model (LLM)</strong> to process research abstracts from the past 
            <strong>5 years</strong> via <strong>OpenAlex</strong> and the <strong>ViCom website</strong>.
            This involved analyzing <strong>75 researchers</strong> and a total of <strong>1,264 publications</strong>. And yes, it’s possible 
            that I missed a few key areas. Still, I hope this gives you a good starting point!
        </ul>

        </p>

        <h5 style="color: #000056;">Final thoughts:</h5>
        <p style="font-size: 14px; line-height: 1.3em; color: #333;">
            If you find this tool helpful and would like to collaborate on improving it (assuming I can carve out some time), 
            feel free to reach out. After all, collaboration is what this is all about, right? 😉
        </p>
        
        <h5 style="color: #000056; text-align: center;">Enjoy exploring your network and happy collaborating! 🚀</h5>
        """,
        unsafe_allow_html=True
    )



    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()



import streamlit as st
import folium
from folium.plugins import MarkerCluster, Fullscreen
from folium import DivIcon
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
from pathlib import Path
import streamlit.components.v1 as components


# -------------------------
# Page Configuration and CSS
# -------------------------
st.set_page_config(page_title="ViCom CollabMap", layout="wide")

st.markdown(
    """
    <style>
    /* Full-width layout and white background */
    body, .reportview-container, .main, .block-container {
        background-color: #ffffff !important;
        color: #333333;
        margin: 0px 0px 0px 0px;
        padding: 10px;
    }
    .reportview-container .main .block-container {
         padding: 0;
         max-width: 100%;
         margin: 0 auto;
    }
    /* Container for collaboration tables: increased left margin and extra vertical spacing */
    .collab-table-container {
        width: 100% !important; 
        margin-left: 0px;
        margin-top: 0px;
    }
    /* Increase minimum width for the third column in dataframes */
    div[data-testid="stDataFrame"] table tr td:nth-child(3) {
         width: 100% !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Data Loading & Utility Functions
# -------------------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def convert_dataframe_types(df):
    for col in df.select_dtypes(include=['int64','float64']).columns:
        if col not in ['Latitude','Longitude']:
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else None)
    return df

def load_data():
    participants_path = BASE_DIR / 'Data_clean' / '01_participants_with_geo.csv'
    projects_path = BASE_DIR / 'Data_clean' / '02_projects.csv'
    potential_collaborations_path = BASE_DIR / 'Data_clean' / '09.potential_collaborations.csv'
    
    participants = pd.read_csv(participants_path)
    projects = pd.read_csv(projects_path)
    potential_collaborations = pd.read_csv(potential_collaborations_path)
    
    projects_geo = projects.merge(
        participants[['full_name','Latitude','Longitude','Affiliation','Photo','Role']],
        on='full_name',
        how='left'
    )
    
    participants = convert_dataframe_types(participants)
    projects_geo = convert_dataframe_types(projects_geo)
    potential_collaborations = convert_dataframe_types(potential_collaborations)
    
    # Remove duplicate researcher pairs
    potential_collaborations = potential_collaborations[
        potential_collaborations['Researcher_A'] < potential_collaborations['Researcher_B']
    ]
    return participants, projects_geo, potential_collaborations

def load_expertise():
    exp_path = BASE_DIR / 'Data_clean' / '08.researchers_with_themes_expertise_cleaned.csv'
    if exp_path.exists():
        expertise_df = pd.read_csv(exp_path)
        expertise_df.columns = [col.strip() for col in expertise_df.columns]
        return expertise_df
    else:
        return pd.DataFrame()

participants, projects_geo, potential_collaborations = load_data()
researchers_expertise = load_expertise()

# -------------------------
# Bézier Curve Function
# -------------------------
def generate_bezier_points(start, end, curvature=0.2, n_points=20, offset=0.001):
    distance = np.linalg.norm(np.array(start) - np.array(end))
    if distance < 1.0:
        curvature = distance * 0.1
    if np.allclose(start, end, atol=0.0001):
        end = [end[0] + offset, end[1] + offset]
    midpoint = [(start[0] + end[0]) / 2 + curvature, (start[1] + end[1]) / 2]
    t_values = np.linspace(0, 1, n_points)
    return [
        (
            (1 - t)**2 * start[0] + 2*(1 - t)*t*midpoint[0] + t**2 * end[0],
            (1 - t)**2 * start[1] + 2*(1 - t)*t*midpoint[1] + t**2 * end[1]
        )
        for t in t_values
    ]

# -------------------------
# Map Helper Functions
# -------------------------
def create_base_map(location=[51,10], zoom=5, tiles="cartodbpositron"):
    m = folium.Map(location=location, zoom_start=zoom, tiles=tiles)
    Fullscreen().add_to(m)
    return m

def generate_project_details_html(researcher_name, filtered_projects):
    rp = filtered_projects[filtered_projects['full_name'] == researcher_name]
    if rp.empty:
        return ""
    unique = set()
    details = ""
    main = rp[rp['Project Type'] == "Main Project"]
    short = rp[rp['Project Type'] == "Short-Term Collaboration"]
    if not main.empty:
        details += '<h6 style="color:blue;">Main Projects:</h6><ul>'
        for _, proj in main.iterrows():
            key = (proj['Project'], proj['Project Type'])
            if key not in unique:
                unique.add(key)
                collabs = ', '.join(filtered_projects[filtered_projects['Project'] == proj['Project']]['full_name'].unique())
                details += f"<li><b>{proj['Project']}</b>: {collabs}</li>"
        details += "</ul>"
    if not short.empty:
        details += '<h6 style="color:green;">Short-Term Collaborations:</h6><ul>'
        for _, proj in short.iterrows():
            key = (proj['Project'], proj['Project Type'])
            if key not in unique:
                unique.add(key)
                collabs = ', '.join(filtered_projects[filtered_projects['Project'] == proj['Project']]['full_name'].unique())
                details += f"<li><b>{proj['Project']}</b>: {collabs}</li>"
        details += "</ul>"
    return details

def create_popup_content(row, project_details):
    role = row['Role'] if pd.notna(row.get('Role')) else "Role not specified"
    
    return f"""
    <div style="text-align:center;">
        <h5>{row['full_name']}</h5>
        <p style="color:gray;">{row['Affiliation']}</p>
        <p><b>Role:</b> {role}</p>
        <div style="margin:auto; width:100px; height:100px; border-radius:50%; overflow:hidden;">
            <img src="{row.get('Photo', 'https://via.placeholder.com/40')}" style="width:100%; height:100%; object-fit:cover;">
        </div>
        
    </div>
    """

def add_researcher_marker(row, filtered_projects, container, highlight=False, border_color="gray", opacity=1.0):
    tooltip_text = f"{row['full_name']} | {row['Affiliation']} | {row.get('Role', 'Role not specified')}"
    details = generate_project_details_html(row['full_name'], filtered_projects)
    popup = create_popup_content(row, details)
    photo_url = row.get('Photo', "https://via.placeholder.com/40")

    # Apply border color and opacity to the marker's icon
    border = f'border:2px solid {border_color};' if highlight else f'border:2px solid {border_color};'
    op_style = f"opacity:{opacity};"
    icon_html = f"""
    <div style="width:40px; height:40px; border-radius:80%; overflow:hidden; {border} {op_style}">
        <img src="{photo_url}" style="width:100%; height:100%; object-fit:cover;">
    </div>
    """
    icon = DivIcon(html=icon_html, icon_size=(10, 10), icon_anchor=(10, 10))
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=folium.Popup(popup, max_width=300),
        tooltip=tooltip_text,
        icon=icon
    ).add_to(container)

def draw_affiliation_lines(filtered_projects, map_obj):
    groups = filtered_projects.groupby("Project")
    seen = set()
    offset = 0.001
    for _, group in groups:
        group = group.dropna(subset=["Latitude", "Longitude"])
        if group.shape[0] < 2:
            continue
        coords = group[['Latitude','Longitude']].values
        project_type = group["Project Type"].iloc[0]
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                start, end = tuple(coords[i]), tuple(coords[j])
                if start == end:
                    continue
                if (start, end) in seen or (end, start) in seen:
                    if project_type == "Main Project":
                        start = (start[0]+offset, start[1]+offset)
                        end = (end[0]-offset, end[1]-offset)
                    elif project_type == "Short-Term Collaboration":
                        start = (start[0]-offset, start[1]-offset)
                        end = (end[0]+offset, end[1]+offset)
                seen.add((start, end))
                bezier = generate_bezier_points(start, end, curvature=0.2)
                color = "blue" if project_type == "Main Project" else "green"
                folium.PolyLine(
                    locations=bezier,
                    color=color,
                    weight=2.5,
                    opacity=0.8
                ).add_to(map_obj)

def add_potential_collaboration_lines(collaborations, map_obj, threshold=0, selected_researcher=None):
    for _, row in collaborations.iterrows():
        if row['Shared_synergies_count'] >= threshold:  # Ensure it includes exactly threshold value
            researcher_a = participants[participants['full_name'] == row['Researcher_A']]
            researcher_b = participants[participants['full_name'] == row['Researcher_B']]
            if researcher_a.empty or researcher_b.empty:
                continue
            start = [researcher_a['Latitude'].values[0], researcher_a['Longitude'].values[0]]
            end = [researcher_b['Latitude'].values[0], researcher_b['Longitude'].values[0]]
            
            if selected_researcher and selected_researcher not in [row['Researcher_A'], row['Researcher_B']]:
                continue
            
            # Calculate saturation based on synergy count
            synergy = row['Shared_synergies_count']
            saturation = min(max((synergy - 1) * 20, 10), 100)  # Scale saturation between 10% and 100%
            color = f'hsl(280, {saturation}%, 50%)'  # Use HSL color with varying saturation
            
            bezier = generate_bezier_points(start, end, curvature=0.2)
            folium.PolyLine(
                locations=bezier,
                color=color,
                weight=2.5,
                opacity=0.8
            ).add_to(map_obj)



def generate_folium_map(participants, projects_geo, potential_collaborations,
                        filter_types=["All"], selected_researcher=None, show_potential=False, synergy_threshold=0):
    default_location = [51, 10]
    zoom_level = 5
    marker_cluster = MarkerCluster(maxClusterRadius=10)
    m = create_base_map(location=default_location, zoom=zoom_level)
    marker_cluster.add_to(m)

    # Initialize collaborators and filtered projects
    collaborators = set()
    potential_collaborators = set()
    filtered_projects = pd.DataFrame()

    if selected_researcher:
        # Filter projects involving the selected researcher
        rp = projects_geo[projects_geo['full_name'] == selected_researcher]

        # Filter by selected project types (Main, Short-Term)
        filtered_projects = projects_geo[
            (projects_geo['Project'].isin(rp['Project'].unique())) &
            (projects_geo['Project Type'].isin(filter_types))
        ]

        # Add collaborators for selected project types
        collaborators = set(filtered_projects['full_name'].unique())

        # Add potential collaborators if enabled and above the threshold
        if show_potential:
            potential_collabs = potential_collaborations[
                ((potential_collaborations['Researcher_A'] == selected_researcher) |
                 (potential_collaborations['Researcher_B'] == selected_researcher)) &
                (potential_collaborations['Shared_synergies_count'] >= synergy_threshold)
            ]
            potential_collaborators = set(
                potential_collabs['Researcher_A'].tolist() + potential_collabs['Researcher_B'].tolist()
            )
            potential_collaborators.discard(selected_researcher)
            collaborators.update(potential_collaborators)

        # Final collaborators set (excluding selected researcher)
        collaborators.discard(selected_researcher)
    else:
        # If no researcher is selected, filter by project type and potential collaborations
        if "All" in filter_types:
            filtered_projects = projects_geo
        else:
            filtered_projects = projects_geo[projects_geo['Project Type'].isin(filter_types)]

        if show_potential:
            potential_collabs = potential_collaborations[
                potential_collaborations['Shared_synergies_count'] >= synergy_threshold
            ]
            potential_collaborators = set(
                potential_collabs['Researcher_A'].tolist() + potential_collabs['Researcher_B'].tolist()
            )
            collaborators.update(potential_collaborators)

    # Add markers for selected researcher and collaborators
    for _, row in participants.iterrows():
        if selected_researcher and row['full_name'] == selected_researcher:
            # Selected researcher marker with red border
            add_researcher_marker(row, filtered_projects, marker_cluster, border_color="red", opacity=1.0)
        elif row['full_name'] in collaborators:
            # Determine border color based on collaboration type
            if row['full_name'] in potential_collaborators:
                border_color = "purple"
            else:
                project_types = filtered_projects[filtered_projects['full_name'] == row['full_name']]['Project Type'].unique()
                if "Main Project" in project_types:
                    border_color = "blue"
                elif "Short-Term Collaboration" in project_types:
                    border_color = "green"
                else:
                    border_color = "gray"
            add_researcher_marker(row, filtered_projects, marker_cluster, border_color=border_color, opacity=0.8)
        elif not selected_researcher:
            # Show all markers if no researcher is selected
            add_researcher_marker(row, filtered_projects, marker_cluster, opacity=0.8)

    # Draw main and short-term collaboration lines if filters are applied
    if not filtered_projects.empty and ("Main Project" in filter_types or "Short-Term Collaboration" in filter_types):
        draw_affiliation_lines(filtered_projects, m)
    
    # Draw potential collaboration lines based on slider threshold
    if show_potential:
        add_potential_collaboration_lines(potential_collaborations, m, threshold=synergy_threshold, selected_researcher=selected_researcher)

    m.save("map.html")
    return m







# -------------------------
# Collaboration Table Functions
# -------------------------
def generate_collaboration_data(selected_researcher):
    # Get the projects the researcher is involved in from projects_geo
    rp = projects_geo[projects_geo['full_name'] == selected_researcher]

    # Group by 'Project' to avoid duplicates and gather necessary data
    grouped = rp.groupby('Project')
    rows = []

    for project_name, group in grouped:
        proj_type = group['Project Type'].iloc[0]  # Get project type (all rows have the same project type)
        
        # Get unique collaborators for the project, excluding the selected researcher
        collabs = projects_geo[(projects_geo['Project'] == project_name) &
                            (projects_geo['full_name'] != selected_researcher)]['full_name'].unique()
        
        # If no collaborators, leave the 'Collaborators' field empty
        rows.append({
            "Project": project_name,
            "Project Type": proj_type,
            "Collaborators": ", ".join(collabs) if len(collabs) > 0 else ""
        })

    # Create dataframe and sort by Project Type (Main Project first)
    existing_df = pd.DataFrame(rows)
    if not existing_df.empty:
        project_type_order = {"Main Project": 0, "Short-Term Collaboration": 1}
        existing_df['Project Type Order'] = existing_df['Project Type'].map(project_type_order)
        existing_df = existing_df.sort_values(by="Project Type Order").drop(columns="Project Type Order").reset_index(drop=True)

    # Generate potential collaborations table
    potential_rows = []
    pot = potential_collaborations[
        (potential_collaborations['Researcher_A'] == selected_researcher) |
        (potential_collaborations['Researcher_B'] == selected_researcher)
    ]
    for _, row in pot.iterrows():
        collaborator = row['Researcher_B'] if row['Researcher_A'] == selected_researcher else row['Researcher_A']
        
        # Get the collaborator's affiliation from the participants dataframe
        affiliation = participants[participants['full_name'] == collaborator]['Affiliation'].values[0]
        
        potential_rows.append({
            "Collaborator": collaborator,
            "Affiliation": affiliation,  # New column for Affiliation
            "Count": row['Shared_synergies_count'],
            "Shared Synergies": row['Shared_synergies']
        })
    
    potential_df = pd.DataFrame(potential_rows)
    if not potential_df.empty:
        potential_df = potential_df.sort_values(by="Count", ascending=False).reset_index(drop=True)
    
    return existing_df, potential_df




# -------------------------
# Streamlit App Layout
# -------------------------
st.sidebar.title("ViCom CollabMap")
st.sidebar.markdown(
    """
    Select a researcher and filter project types using the checkboxes.
    """,
    unsafe_allow_html=True,
)
selected_researcher = st.sidebar.selectbox(
    "Select a Researcher:",
    [""] + sorted(participants['full_name'].unique())
)

# Place researcher expertise after checkboxes as the last sidebar filter.
main_project = st.sidebar.checkbox("🔵 Main Project", value=True, help="Main ViCom project")
short_term = st.sidebar.checkbox("🟢 Short-Term Collaboration", value=True, help="Short-Term collaborations provide extra training and foster new research networks")
filter_types = []
if main_project:
    filter_types.append("Main Project")
if short_term:
    filter_types.append("Short-Term Collaboration")

show_potential = st.sidebar.checkbox("🟣 Potential Collaborations", value=False, help="Potential collaborations based on research area/expertise in the last 5 years")
synergy_threshold = 0
if show_potential:
    if selected_researcher:
        # Get the maximum synergy count for the selected researcher
        max_synergy = potential_collaborations[
            (potential_collaborations['Researcher_A'] == selected_researcher) |
            (potential_collaborations['Researcher_B'] == selected_researcher)
        ]['Shared_synergies_count'].max()

        if pd.notna(max_synergy):  # Ensure max_synergy is not NaN
            synergy_threshold = st.sidebar.slider(
                "Minimum Shared Synergy Count",
                min_value=0,
                max_value=int(max_synergy),
                value=0,
                step=1
            )
    else:
        max_synergy = int(potential_collaborations['Shared_synergies_count'].max())
        synergy_threshold = st.sidebar.slider(
            "Minimum Shared Synergy Count",
            min_value=0,
            max_value=max_synergy,
            value=0,
            step=1
        )

    st.sidebar.markdown(
        "<p style='color:purple;'>Purple line intensity reflects synergy count.</p>",
        unsafe_allow_html=True
    )

if selected_researcher:
    exp = researchers_expertise[researchers_expertise["Full Name"] == selected_researcher]
    if not exp.empty:
        st.sidebar.expander(f"Show {selected_researcher} expertise").markdown(
        f"""
        <div style="font-size: 12px;">
            <b>Themes:</b> {exp['Themes'].values[0]}<br><br>
            <b>Expertise:</b> {exp['Expertise'].values[0]}
        </div>
        """,
        unsafe_allow_html=True
    )

# Main Map Area – embed the fully interactive Folium map using components.html.
map_object = generate_folium_map(participants, projects_geo, potential_collaborations,
                                 filter_types=filter_types,
                                 selected_researcher=selected_researcher if selected_researcher != "" else None,
                                 show_potential=show_potential,
                                 synergy_threshold=synergy_threshold)
components.html(map_object.get_root().render(), height=500)

# Collaboration Tables Section
if selected_researcher:
    existing_df, potential_df = generate_collaboration_data(selected_researcher)
    with st.container():
        st.markdown("<div class='collab-table-container'>", unsafe_allow_html=True)
        st.markdown("**Projects**", unsafe_allow_html=True)
        if not existing_df.empty:
            def style_existing(row):
                if row["Project Type"] == "Main Project":
                    return ['background-color: #cce5ff'] * len(row)
                else:
                    return ['background-color: #d4edda'] * len(row)
            dynamic_height = max(100, 30 * (len(existing_df) + 1))

            st.dataframe(
                existing_df.style.set_properties(**{'font-size': '10px', 'font-weight': 'bold'})
                .apply(style_existing, axis=1)
                .hide(axis="index"),
                height=100,
                use_container_width=True
            )
        else:
            st.info("No existing collaborations found.")
        st.markdown("**Potential Collaborations**", unsafe_allow_html=True)
        if not potential_df.empty:
            def style_potential(row):
                synergy = row["Count"]
                sat = min(max((synergy - 1) * 20, 10), 100)
                return ['background-color: ' + f"hsl(280, {sat}%, 95%)"] * len(row)
            st.dataframe(potential_df.style.set_properties(**{'font-size': '10px', 'font-weight': 'bold'}).apply(style_potential, axis=1).hide(axis="index"), height=500, use_container_width=True)
        else:
            st.info("No potential collaborations found.")
            
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Select a researcher from the sidebar to see collaboration details.")

address = st.get_option("server.address") or "localhost"
port = st.get_option("server.port") or "8501"
url = f"http://{address}:{port}"
