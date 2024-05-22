from django.contrib import admin
from .models import Image, Tag, Comment

admin.site.register(Image)
admin.site.register(Tag)
admin.site.register(Comment)