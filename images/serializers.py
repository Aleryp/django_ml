from rest_framework import serializers
from .models import Image, Comment, Tag

class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = ['id', 'name']

class CommentSerializer(serializers.ModelSerializer):
    user = serializers.ReadOnlyField(source='user.username')

    class Meta:
        model = Comment
        fields = ['id', 'user', 'text', 'created_at']

class ImageSerializer(serializers.ModelSerializer):
    user = serializers.ReadOnlyField(source='user.username')
    comments = CommentSerializer(many=True, read_only=True)
    tags = TagSerializer(many=True, read_only=True)

    class Meta:
        model = Image
        fields = ['id', 'user', 'image', 'description', 'tags', 'comments', 'created_at']
