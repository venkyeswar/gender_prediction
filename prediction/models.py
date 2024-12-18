from django.db import models

class Image(models.Model):
    image = models.ImageField(upload_to='uploads/')  # Save uploaded images to the 'uploads' folder

    def __str__(self):
        return self.image.name
