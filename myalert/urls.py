from django.contrib.auth import views as auth_views
from django.conf.urls import url
from django.contrib import admin
from django.urls import path, include
from . import views

app_name = 'myalert'

urlpatterns = [
    url(r"^login/$", auth_views.LoginView.as_view(template_name="login.html"), name='login'),
    url(r"^logout/$", auth_views.LogoutView.as_view(), name="logout"),
    url(r"^signup/$", views.SignUp.as_view(), name="signup"),
    url(r"^drive/$", views.Drive, name="drive"),
    url(r"^addprofile/$", views.add_profile.as_view(), name="addprofile"),
    url(r"^mypro/$", views.get_info, name="mypro"),

]
