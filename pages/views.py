import mysql.connector
from django.shortcuts import render

import os
from pathlib import Path

import json
import pandas as pd

from advise_agent.advise import main, advise_he, Advise


def home_view(request, *args, **kwargs):
    path = os.path.join('home_page.html')
    return render(request, path, {})


def query(q):
    BASE_DIR = Path(__file__).resolve().parent.parent

    with open(os.path.join(BASE_DIR, 'cw_site', 'security', 'security.json'), 'r') as f:
        data = json.load(f)

    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password=data["password"],
        database='COURSEWORK'
    )

    cursor = mydb.cursor()
    cursor.execute(q)
    result = cursor.fetchall()
    return result


def search_view(request, *args, **kwargs):
    concept = request.GET.get('concept_search')
    path = os.path.join('search.html')
    if concept == '':
        return render(request, path, {})
    select = f"""SELECT concept, description FROM concepts
                 WHERE concept LIKE '%{concept}%';"""
    try:
        c = query(select)
        concepts = {c[i][0]: c[i][1] for i in range(len(c))}
        context = {
            "concepts": concepts
        }
        return render(request, path, context)
    except Exception:
        return render(request, path, {})


def advise_he_form_view(request, *args, **kwargs):
    path = os.path.join('he_form.html')
    return render(request, path, {})


def advise_he_view(request, *args, **kwargs):
    preffered = request.POST.getlist('preferred')
    profiles = ["avia",
        "bio",
        "vet",
        "war",
        "geo",
        "design",
        "journ",
        "engin",
        "it",
        "art",
        "hist",
        "forest",
        "lingo",
        "math",
        "med",
        "int_com",
        "music",
        "teach",
        "psy",
        "com",
        "agri",
        "ss",
        "theo",
        "techn",
        "technl",
        "turism",
        "physics",
        "phys_cult",
        "chem",
        "econ",
        "law"]

    context = {profile: [(1 if profile in preffered else None)] for profile in profiles}
    context.update({"region": request.POST.get('region'),
                    "dormitory": request.POST.get('dormitory')})
    context_pd = pd.DataFrame(data=context)

    model = request.POST.get('model')
    result = predict_he(context_pd, model)

    path = os.path.join('he_advise.html')
    return render(request, path, {"he": result[0]})


def predict_he(to_predict, preferred):
    adv = Advise()
    main(adv, preferred)
    result = advise_he(to_predict, adv)
    return result


if __name__ == '__main__':
    pass

