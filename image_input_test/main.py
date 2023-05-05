from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserListView
from kivy.core.window import Window

class ImageInput(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        
        # Create an image widget to display the selected image
        self.image = Image(source='', size_hint=(1, 1), allow_stretch=True)
        self.add_widget(self.image)
        
        # Create a file chooser widget to allow the user to browse for an image
        self.file_chooser = FileChooserListView(
            size_hint=(1, 0.2), 
            path=App.get_running_app().user_data_dir, 
            filters=['*.png', '*.jpg', '*.jpeg']
        )
        self.file_chooser.bind(selection=self.load_image)
        self.add_widget(self.file_chooser)
        
        # Enable drag and drop on the image widget
        self.image.allow_drop = True
        self.image.bind(on_dropfile=self.load_image)
        
    def load_image(self, instance, value):
        # Set the source of the image widget to the selected file
        if isinstance(value, list):
            filename = value[0]
        else:
            filename = value
        self.image.source = filename

class MyApp(App):
    def build(self):
        return ImageInput()

if __name__ == '__main__':
    MyApp().run()