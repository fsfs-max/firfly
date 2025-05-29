import os
import random

from kivy.app import App
from kivy.clock import Clock
from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivy.properties import ListProperty, ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.textinput import TextInput

# 提前注册字体
current_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(current_dir, 'SimHei.ttf')
LabelBase.register(
    name='SimHei',
    fn_regular=font_path,
    fn_bold=font_path,
    fn_italic=font_path,
    fn_bolditalic=font_path
)


# 模拟食材识别结果
class MockFoodRecognizer:
    def __init__(self):
        self.sample_ingredients = [
            "西红柿", "鸡蛋", "土豆", "青椒", "胡萝卜", "洋葱", "大蒜", "生姜",
            "鸡肉", "牛肉", "猪肉", "米饭", "面粉", "食用油", "盐", "糖"
        ]

    def recognize(self, image_path=None):
        return random.sample(self.sample_ingredients, random.randint(3, 6))


# 模拟食谱推荐系统
class MockRecipeRecommender:
    def __init__(self):
        self.recipes = self._load_recipes()

    def _load_recipes(self):
        recipes = [
            {
                "id": 1,
                "name": "番茄炒蛋",
                "ingredients": ["西红柿", "鸡蛋", "盐", "糖", "食用油"],
                "steps": [
                    "将西红柿洗净切块，鸡蛋打散备用",
                    "锅中倒油，油热后倒入鸡蛋液，炒熟盛出",
                    "锅中再倒少许油，放入西红柿块翻炒",
                    "加入适量盐和糖调味",
                    "最后倒入炒好的鸡蛋，翻炒均匀即可"
                ],
                "image": "https://picsum.photos/seed/recipe1/400/300"
            },
            {
                "id": 2,
                "name": "土豆烧牛肉",
                "ingredients": ["土豆", "牛肉", "洋葱", "大蒜", "生姜", "盐", "食用油"],
                "steps": [
                    "将牛肉切块，焯水去血沫",
                    "土豆去皮切块，洋葱、大蒜、生姜切碎",
                    "锅中倒油，放入洋葱、大蒜、生姜爆香",
                    "加入牛肉块翻炒，然后加入适量清水",
                    "大火烧开后转小火慢炖1小时",
                    "加入土豆块继续炖20分钟，直到土豆软烂",
                    "最后加盐调味即可"
                ],
                "image": "https://picsum.photos/seed/recipe2/400/300"
            },
            {
                "id": 3,
                "name": "青椒土豆丝",
                "ingredients": ["土豆", "青椒", "盐", "食用油", "大蒜"],
                "steps": [
                    "将土豆去皮切成细丝，用清水冲洗掉淀粉",
                    "青椒切丝，大蒜切末",
                    "锅中倒油，油热后放入大蒜末爆香",
                    "加入土豆丝翻炒",
                    "加入青椒丝继续翻炒",
                    "最后加盐调味，翻炒均匀即可"
                ],
                "image": "https://picsum.photos/seed/recipe3/400/300"
            },
            {
                "id": 4,
                "name": "胡萝卜炒鸡蛋",
                "ingredients": ["胡萝卜", "鸡蛋", "盐", "食用油"],
                "steps": [
                    "将胡萝卜洗净切丝，鸡蛋打散备用",
                    "锅中倒油，油热后倒入鸡蛋液，炒熟盛出",
                    "锅中再倒少许油，放入胡萝卜丝翻炒",
                    "加入适量盐调味",
                    "最后倒入炒好的鸡蛋，翻炒均匀即可"
                ],
                "image": "https://picsum.photos/seed/recipe4/400/300"
            },
            {
                "id": 5,
                "name": "红烧肉",
                "ingredients": ["猪肉", "生姜", "大蒜", "冰糖", "酱油", "料酒", "食用油"],
                "steps": [
                    "将猪肉切成方块，焯水去血沫",
                    "锅中倒油，放入冰糖炒出糖色",
                    "加入猪肉块翻炒上色",
                    "加入生姜、大蒜、酱油、料酒继续翻炒",
                    "加入适量清水，大火烧开后转小火慢炖1小时",
                    "最后收汁即可"
                ],
                "image": "https://picsum.photos/seed/recipe5/400/300"
            }
        ]
        return recipes

    def recommend(self, ingredients):
        recommended = []
        for recipe in self.recipes:
            matched = sum(1 for ing in ingredients if ing in recipe["ingredients"])
            if matched >= len(ingredients) / 2:
                recipe = recipe.copy()
                recipe["match_score"] = matched / len(recipe["ingredients"])
                recommended.append(recipe)

        recommended.sort(key=lambda x: x["match_score"], reverse=True)
        return recommended


# 拍照界面
class CameraScreen(Screen):
    camera = ObjectProperty(None)
    captured_texture = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(CameraScreen, self).__init__(**kwargs)
        self.food_recognizer = MockFoodRecognizer()

    def capture(self):
        loading_popup = Popup(
            title="处理中",
            title_font='SimHei',  # 关键修改：指定标题字体
            content=Label(text="正在识别食材，请稍候...", font_name='SimHei'),
            size_hint=(None, None),
            size=(300, 200)
        )
        loading_popup.open()
        Clock.schedule_once(lambda dt: self._process_ingredients(loading_popup), 2)

    def _process_ingredients(self, popup):
        popup.dismiss()
        ingredients = self.food_recognizer.recognize("captured_image.png")
        confirm_screen = self.manager.get_screen('confirm')
        confirm_screen.set_ingredients(ingredients)
        self.manager.transition.direction = 'left'
        self.manager.current = 'confirm'


# 食材确认界面
class ConfirmScreen(Screen):
    ingredients = ListProperty([])
    ingredient_grid = ObjectProperty(None)

    def set_ingredients(self, ingredients):
        self.ingredients = ingredients
        self.update_ingredient_list()

    def update_ingredient_list(self):
        self.ingredient_grid.clear_widgets()
        for ingredient in self.ingredients:
            ingredient_item = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
            ingredient_item.add_widget(Label(text=ingredient, font_name='SimHei'))

            remove_btn = Button(text='删除', size_hint_x=0.3, font_name='SimHei')
            remove_btn.bind(on_press=lambda instance, ing=ingredient: self.remove_ingredient(ing))
            ingredient_item.add_widget(remove_btn)
            self.ingredient_grid.add_widget(ingredient_item)

    def remove_ingredient(self, ingredient):
        if ingredient in self.ingredients:
            self.ingredients.remove(ingredient)
            self.update_ingredient_list()

    def add_ingredient(self):
        popup_content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        ingredient_input = TextInput(
            hint_text='输入食材名称',
            multiline=False,
            font_name='SimHei'
        )
        button_layout = BoxLayout(size_hint_y=0.3, spacing=10)

        add_button = Button(text='添加', font_name='SimHei')
        cancel_button = Button(text='取消', font_name='SimHei')

        button_layout.add_widget(add_button)
        button_layout.add_widget(cancel_button)

        popup_content.add_widget(ingredient_input)
        popup_content.add_widget(button_layout)

        popup = Popup(
            title='添加食材',
            title_font='SimHei',
            content=popup_content,
            size_hint=(None, None),
            size=(400, 200)
        )

        def on_add(instance):
            ingredient = ingredient_input.text.strip()
            if ingredient and ingredient not in self.ingredients:
                self.ingredients.append(ingredient)
                self.update_ingredient_list()
            popup.dismiss()

        def on_cancel(instance):
            popup.dismiss()

        add_button.bind(on_press=on_add)
        cancel_button.bind(on_press=on_cancel)
        popup.open()

    def confirm_ingredients(self):
        if not self.ingredients:
            popup = Popup(
                title='提示',
                content=Label(text='请至少选择一种食材', font_name='SimHei'),
                size_hint=(None, None),
                size=(300, 200)
            )
            popup.open()
            return

        loading_popup = Popup(
            title="推荐中",
            title_font='SimHei',
            content=Label(text="正在为您推荐食谱，请稍候...", font_name='SimHei'),
            size_hint=(None, None),
            size=(300, 200)
        )
        loading_popup.open()
        Clock.schedule_once(lambda dt: self._recommend_recipes(loading_popup), 2)

    def _recommend_recipes(self, popup):
        popup.dismiss()
        recipe_recommender = MockRecipeRecommender()
        recipes = recipe_recommender.recommend(self.ingredients)

        if not recipes:
            popup = Popup(
                title='提示',
                title_font='SimHei',
                content=Label(text='没有找到匹配的食谱，请尝试添加更多食材', font_name='SimHei'),
                size_hint=(None, None),
                size=(300, 200)
            )
            popup.open()
            return

        recipe_screen = self.manager.get_screen('recipe_list')
        recipe_screen.set_recipes(recipes)
        self.manager.transition.direction = 'left'
        self.manager.current = 'recipe_list'


# 食谱列表界面
class RecipeListScreen(Screen):
    recipes = ListProperty([])
    recipe_grid = ObjectProperty(None)

    def set_recipes(self, recipes):
        self.recipes = recipes
        self.update_recipe_list()

    def update_recipe_list(self):
        self.recipe_grid.clear_widgets()
        for recipe in self.recipes:
            card = BoxLayout(orientation='vertical', size_hint_y=None, height=200,
                             spacing=10, padding=10, pos_hint={'center_x': 0.5})
            card.border = [10, 10, 10, 10]

            img = Image(source=recipe['image'], size_hint_y=0.7)
            card.add_widget(img)

            info_layout = BoxLayout(orientation='horizontal', size_hint_y=0.3)
            name_label = Label(
                text=recipe['name'],
                font_name='SimHei',
                size_hint_x=0.8,
                halign='left'
            )
            match_label = Label(
                text=f"匹配度: {recipe['match_score']:.0%}",
                size_hint_x=0.2,
                color=(0, 0.5, 0, 1),
                font_name='SimHei'
            )

            info_layout.add_widget(name_label)
            info_layout.add_widget(match_label)
            card.add_widget(info_layout)

            card.bind(on_touch_down=lambda instance, touch, r=recipe:
            self.show_recipe_detail(r) if instance.collide_point(*touch.pos) else False)

            self.recipe_grid.add_widget(card)

    def show_recipe_detail(self, recipe):
        detail_screen = self.manager.get_screen('recipe_detail')
        detail_screen.set_recipe(recipe)
        self.manager.transition.direction = 'left'
        self.manager.current = 'recipe_detail'


# 食谱详情界面
class RecipeDetailScreen(Screen):
    recipe = ObjectProperty(None)
    recipe_image = ObjectProperty(None)
    recipe_name = ObjectProperty(None)
    ingredients_text = ObjectProperty(None)
    steps_text = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(RecipeDetailScreen, self).__init__(**kwargs)
        # 设置内边距
        self.padding = [20, 20, 20, 20]

    def set_recipe(self, recipe):
        self.recipe = recipe
        self.update_recipe_detail()

    def update_recipe_detail(self):
        if not self.recipe:
            return

        self.recipe_image.source = self.recipe['image']
        self.recipe_name.text = self.recipe['name']
        self.recipe_name.font_name = 'SimHei'

        # 处理食材列表
        ingredients_text = "所需食材:\n" + "\n".join([f"- {ingredient}" for ingredient in self.recipe['ingredients']])
        self.ingredients_text.text = ingredients_text
        self.ingredients_text.font_name = 'SimHei'
        self.ingredients_text.text_size = (self.width - 40, None)  # 宽度减去内边距
        self.ingredients_text.multiline = True
        self.ingredients_text.valign = 'top'

        # 处理步骤列表（关键修复）
        steps_text = "烹饪步骤:\n" + "\n\n".join([f"{i}. {step}" for i, step in enumerate(self.recipe['steps'], 1)])
        self.steps_text.text = steps_text
        self.steps_text.font_name = 'SimHei'
        self.steps_text.text_size = (self.width - 40, None)  # 设置文本显示宽度
        self.steps_text.multiline = True  # 必须开启，否则无法换行
        self.steps_text.valign = 'top' # 确保步骤字体


# 主应用
class RecipeApp(App):
    def build(self):
        sm = ScreenManager(transition=SlideTransition())
        sm.add_widget(CameraScreen(name='camera'))
        sm.add_widget(ConfirmScreen(name='confirm'))
        sm.add_widget(RecipeListScreen(name='recipe_list'))
        sm.add_widget(RecipeDetailScreen(name='recipe_detail'))
        sm.current = 'camera'
        return sm


if __name__ == '__main__':
    Window.size = (400, 600)
    RecipeApp().run()