from rest_framework import viewsets
from django.contrib.auth.models import User

from .models import UserProfile
from .serializers import UserSerializer, UserProfileSerializer
from rest_framework import generics

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer


class UserProfileViewSet(viewsets.ModelViewSet, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = UserProfileSerializer
    queryset = UserProfile.objects.all()
    basename = 'userprofile'

    def get_queryset(self):
        queryset = super().get_queryset()
        user_id = self.request.user.id
        if user_id is not None:
            queryset = queryset.filter(user_id=user_id)
        return queryset

    def get_object(self):
        queryset = super().get_queryset()
        user_id = self.request.user.id
        if user_id is not None:
            obj = queryset.filter(user_id=user_id)[0]

        return obj

    def partial_update(self, request, *args, **kwargs):
        kwargs['partial'] = True
        if request.data.get('email'):
            user_id = self.request.user.id
            if user_id is not None:
                obj = User.objects.get(id=user_id)
                obj.email = request.data.get('email')
                print(request.data.get('email'))
                obj.save()
        return self.update(request, *args, **kwargs)
