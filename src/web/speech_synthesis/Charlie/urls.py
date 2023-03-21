from django.urls import path
from Charlie import views
from django.views.decorators.csrf import csrf_exempt

app_name = 'Charlie'


urlpatterns = [
    path('', views.Index.as_view(), name='index'),
    path('about/', views.AboutView.as_view(), name='about'),
    path('register_profile/', views.RegisterProfileView.as_view(),
         name='register_profile'),
    path('profile/<username>/', views.ProfileView.as_view(), name='profile'),
    path('profiles/', views.ListProfilesView.as_view(), name='profiles'),
    path('synthesizer/', views.SynthesizerView.as_view(), name='synthesizer'),
    path('samples/', views.SampleView.as_view(), name='sample'),
    path('loader/', views.LoaderView.as_view(), name='loader'),
]