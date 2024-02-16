from st_pages import Page, show_pages, add_page_title
import json


emoji = json.load(open('Работа_1/Data/emoji.json'))

add_page_title()
show_pages(
    [
        Page("Работа_1/Page/RegressionModel.py", "Модель регресиии", emoji["green_book"]),

        Page("Работа_1/Page/ClassificationModel.py", "Модель классификации", emoji['blue_book']),
    ]
)
