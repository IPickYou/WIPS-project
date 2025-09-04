import streamlit as st

def show():

    default_categories = {
        "CPC_C01B" : "Non-metal elements and their compounds (excluding CO2). Inorganic compounds without metals.",
        "CPC_C01C" : "Ammonia, cyanide, and their compounds.",
        "CPC_C01D" : "Alkali metal compounds such as lithium, sodium, potassium, rubidium, cesium, or francium.",
        "CPC_C01F" : "Compounds of metals like beryllium, magnesium, aluminum, calcium, strontium, barium, radium, thorium, or rare earth metals.",
        "CPC_C01G" : "Compounds containing metals not included in C01D or C01F."
    }
    
    if 'categories' not in st.session_state:
        st.session_state.categories = default_categories.copy()
    
    if 'num_categories' not in st.session_state:
        st.session_state.num_categories = len(st.session_state.categories)
    
    with st.expander("**CATEGORY TO CLASSIFY**", expanded = False):

        col1, col2 = st.columns([1, 3.19])

        with col1:
            st.write("**CATEGORY**")

        with col2:
            st.write("**DESCRIPTION**")
    
        updated_categories = {}
        categories_to_remove = []
        
        category_items = list(st.session_state.categories.items())
        
        for i in range(st.session_state.num_categories):

            col1, col2, col3 = st.columns([1, 3, 0.15])

            default_key = category_items[i][0] if i < len(category_items) else f"CPC_CODE_{i+1}"
            default_value = category_items[i][1] if i < len(category_items) else ""
            
            with col1:
                code = st.text_input(f"CATEGORY {i + 1}", value=default_key, key = f"code_{i}")
            with col2:
                desc = st.text_area(f"DESCRIPTION {i + 1}", value=default_value, key = f"desc_{i}", height = 30)
            with col3:
                if st.button("✖️", key = f"remove_{i}", help = "삭제"):
                    if st.session_state.num_categories > 1:
                        categories_to_remove.append(i)
            
            if code.strip() and desc.strip():
                updated_categories[code.strip()] = desc.strip()
        
        if categories_to_remove:

            st.session_state.num_categories -= len(categories_to_remove)

            remaining_items = []

            for i, (key, value) in enumerate(category_items):
                if i not in categories_to_remove:
                    remaining_items.append((key, value))
            
            st.session_state.categories = dict(remaining_items)

            st.rerun()

        st.session_state.categories = updated_categories
        
        col1, col2 = st.columns([1, 5.5])

        with col1:
            if st.button("➕ ADD CATEGORY"):
                st.session_state.num_categories += 1
                st.rerun()
        
        with col2:
            if st.button("®️ RESET TO DEFAULT", key = "reset_categories_btn"):
                st.session_state.categories = default_categories.copy()
                st.session_state.num_categories = len(default_categories)
                st.rerun()