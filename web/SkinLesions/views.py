from django.shortcuts import render


# Information relative to the dissertation objectives and purpose.
def about(request):
    return render(request, 'about.html')


def base_layout(request):
    return render(request, 'layout.html')
