from django import forms
from Charlie.models import UserProfile, Synthesizer
from django.contrib.auth.models import User


class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput())

    class Meta:
        model = User
        fields = ('username', 'email', 'password')


class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ('website', 'picture')


class SynthesizerForm(forms.ModelForm):
    class Meta:
        model = Synthesizer
        fields = ('text', 'input_audio')