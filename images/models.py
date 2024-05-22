from django.conf import settings
from django.db import models
from django.contrib.auth.models import User

from src.utils import generate_tags


class Tag(models.Model):
    name = models.CharField(max_length=50, unique=True)

    def __str__(self):
        return self.name


class Image(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='images')
    description = models.TextField()
    tags = models.ManyToManyField(Tag, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def save(
            self, force_insert=False, force_update=False, using=None, update_fields=None
    ):
        super().save(force_insert=False, force_update=False, using=None, update_fields=None)
        tags = generate_tags(self.image.path)
        for tag_name in tags:
            tag, created = Tag.objects.get_or_create(name=tag_name)
            self.tags.add(tag)

class Comment(models.Model):
    image = models.ForeignKey(Image, related_name='comments', on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
