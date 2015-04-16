from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader, Context
from django.views.generic.base import TemplateView
from forms import FilesForm, LoginForm, UserForm
from models import SongInterface
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import Permission, User
from django.contrib.auth import logout
from django.template.context_processors import csrf
from django.shortcuts import render_to_response
from software.CognitiveController import *
def logout_view(request):
    logout(request)
    return loginPage(request)
def index(request):
    context = { "form" : FilesForm() }
    html = render(request, 'base.html', context)
    return HttpResponse(html)
def fileUpload(request):
    print request.POST
    x =  request.FILES['file']
    print x
    print type(x)
    print x.content_type
    if x.content_type != "audio/x-wav":
        context = { "form" : FilesForm() }
        html = render(request, 'base.html', context)
        return render_to_response(html)

    if request.method == 'POST':
        user = request.user
        print user
        songToUpload = SongInterface(title=request.FILES['file'].name ,fileIn =request.FILES['file'],jobOwner = request.user)
        if songToUpload.fileIn != None and songToUpload.title != None:
             songToUpload.save()
             template_name = "home.html"
             c = {}
             c.update(csrf(request))
             html = loader.get_template(template_name)
             return render_to_response(template_name,c)
        else:
            context = { "form" : FilesForm() }
            html = render(request, 'base.html', context)
            return render_to_response(html)
def loginPage(request):
        if request.user.is_authenticated():
            return index(request)
        context = { "form" : LoginForm() }
        html = render(request, 'login.html', context)
        return HttpResponse(html)
def loginAttempt(request):
    print request.POST
    username = request.POST['emailAddress']
    password = request.POST['password']
    user = authenticate(username=username, password=password)
    if user is not None:
        if user.is_active:
            login(request, user)
            return index(request)
        else:
            return loginPage(request)
    else:
        return loginPage(request)
def signUp(request):
  print request.POST
  username = request.POST['emailAddress']
  password = request.POST['password']
  lform = UserForm(request.POST)
  if lform.is_valid():
        new_user = User.objects.create_user(username=username,password=password)
        new_user.save()
        user = authenticate(username=username,password=password)
        login(request,user)
        print new_user
        return index(request)
  else:
    return loginPage(request)
def runJob(request):
    user = request.user
    x = SongInterface.objects.filter(jobOwner = user)
    return jobFinished(request)

def jobFinished(request):
        results = getRandomResults()
        context = { "results" : results }
        html = render(request, 'results.html', context) 
        return html
class HomeView(TemplateView):

    template_name = "home.html"

    def get_context_data(self, **kwargs):
        context = super(HomeView, self).get_context_data(**kwargs)
        return context