from django.contrib.auth.models import User
from django.db import models
from django.dispatch import receiver
from django.contrib.auth import get_user_model


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    bio = models.TextField(blank=True, null=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)


@receiver(models.signals.post_save, sender=get_user_model())
def create_student_after_user_init(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance, bio='', avatar='')