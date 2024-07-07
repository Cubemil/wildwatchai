import streamlit as st
import random


# def gamification():
#     col0, col1, col2 = st.columns([0.1, 0.8, 0.1], gap="small")
#     with col1:
#         st.image(st.session_state.current_image, use_column_width=True)
#         render_radio()
#
#         if st.button(
#                 "Lösung abgeben",
#                 disabled=st.session_state.solve_button_disabled
#         ):
#             solve_button()
#
#         if st.button(
#                 "Weiter",
#                 disabled=st.session_state.continue_button_disabled
#         ):
#             continue_button()
#
#         st.text(f"Punktzahl: {st.session_state.score}")
#         if st.session_state.game_over:
#             st.text("GAME OVER")


def render_ai():
    st.image(st.session_state.ai_image, use_column_width=True)


def render_radio():
    return st.radio(
        label="Welches Tier ist zu erkennen?",
        options=["Feldhase", "Feldmaus", "Fledermaus", "Fuchs", "Reh", "Waschbär", "Wildschwein"],
        index=st.session_state.index,
        disabled=False,
        key="spotted_animal",
        on_change=radio_on_change()
    )


def radio_on_change() -> None:
    selected_radio = st.session_state.spotted_animal
    transformed = transform_radio_return_to_index(selected_radio)

    if transformed is not None:
        st.session_state.spotted_animal = selected_radio
        st.session_state.index = transformed


def solve_button():
    r = solve(st.session_state.spotted_animal, st.session_state.ai_result)
    st.session_state.index = None
    st.session_state.continue_button_disabled = False
    st.session_state.solve_button_disabled = True
    #st.rerun()


def continue_button():
    if len(st.session_state.available_images) != 0:
        st.session_state.current_image = st.session_state.images_files[random_chose_image() - 1]
        st.session_state.continue_button_disabled = True
        st.session_state.solve_button_disabled = False
        st.session_state.index = None
        st.rerun()
    else:
        st.session_state.continue_button_disabled = True
        st.session_state.solve_button_disabled = True
        st.session_state.game_over = True
        st.rerun()


def load_session_state():
    if "index" not in st.session_state:
        st.session_state.index = None
    if "spotted_animal" not in st.session_state:
        st.session_state.spotted_animal = None
    if "available_images" not in st.session_state:
        st.session_state.available_images = [1, 2, 3, 4, 5]
    if "images_files" not in st.session_state:
        st.session_state.images_files = ["streamlit/images/feldhase1.jpg",
                                         "streamlit/images/fuchs1.jpg",
                                         "streamlit/images/reh1.jpg",
                                         "streamlit/images/wildschwein1.jpg",
                                         "streamlit/images/wildschwein2.jpg"
                                         ]
    if "current_image" not in st.session_state:
        st.session_state.current_image = st.session_state.images_files[random_chose_image() - 1]
    if "solve_button_disabled" not in st.session_state:
        st.session_state.solve_button_disabled = False
    if "continue_button_disabled" not in st.session_state:
        st.session_state.continue_button_disabled = True
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "game_over" not in st.session_state:
        st.session_state.game_over = False
    if "ai_result" not in st.session_state:
        st.session_state.ai_result = ""


def random_chose_image():
    index = random.randint(0, len(st.session_state.available_images) - 1)
    num = st.session_state.available_images[index]
    st.session_state.available_images.pop(index)
    return num


def transform_radio_return_to_index(option: str):
    if option == "Feldhase":
        return 0
    elif option == "Feldmaus":
        return 1
    elif option == "Fledermaus":
        return 2
    elif option == "Fuchs":
        return 3
    elif option == "Reh":
        return 4
    elif option == "Waschbär":
        return 5
    elif option == "Wildschwein":
        return 6
    else:
        return None


def is_animal(choice: str) -> bool:
    animal = st.session_state.current_image
    ind_slash = animal.rfind('/')
    ind_dot = animal.rfind('.')
    animal = animal[ind_slash + 1:ind_dot].strip('0123456789')

    if choice == animal:
        return True
    else:
        return False


def solve(choice: str, ai_choice: str):

    c = str(choice).lower()
    ai = str(ai_choice).lower()
    if is_animal(c) and not is_animal(ai):
        st.session_state.score += 3
    elif is_animal(c) and is_animal(ai):
        st.session_state.score += 1
    else:
        st.session_state.score += 0

