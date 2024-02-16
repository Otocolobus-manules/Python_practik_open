import streamlit as stm
from PIL import Image


def base_param(link_image="Работа_1/Data/images/logo.png", layout='centered',
               initial_sidebar_state="auto", page_title="App_page", menu_items=None) -> None:
    """Задает базовые параметры страницы"""
    parameters = {
        'layout': layout,
        'initial_sidebar_state': initial_sidebar_state,
        'page_title': page_title,
        'menu_items': menu_items,
    }
    image_directory = link_image
    icon = Image.open(image_directory)

    parameters['page_icon'] = icon
    stm.set_page_config(**parameters)
