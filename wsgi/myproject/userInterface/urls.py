from django.conf.urls import url
from userInterface.views import HomeView
from views import index, fileUpload, loginPage, loginAttempt, signUp, logout_view, runJob, jobFinished
from forms import NameForm

urlpatterns = (
    url(r'^$', index,name = "index"),
    url(r"fileUpload/", fileUpload, name="fileUpload"),
    url(r"loginPage/", loginPage, name="loginPage"),
    url(r"loginAttempt/", loginAttempt, name="loginAttempt"),
    url(r"signUp/", signUp, name="signUp"),
    url(r"logout_view/", logout_view, name="logout_view"),
    url(r"runJob/", runJob, name="runJob"),
    url(r"jobFinished/", jobFinished, name="jobFinished"),
)