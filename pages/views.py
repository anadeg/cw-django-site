import mysql.connector
from django.shortcuts import render

import os
from pathlib import Path

import json
import pandas as pd

from advise_agent.advise import main, advise_he, Advise
from sklearn.metrics import accuracy_score


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
    concept = request.GET.get('concept_search') or 'социальный институт'
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


def search_base_inclusions_view(request, concept, *args, **kwargs):
    path = os.path.join('inclusions.html')
    select = f"""select distinct c1.concept, c2.concept, c2.description
                from concepts as c1
                inner join inclusions
                on c1.id = inclusions.parentid
                inner join concepts as c2
                on inclusions.childid = c2.id
                where c1.concept = '{concept}';"""
    try:
        c = query(select)
        incl_descr = {c[i][1]: c[i][-1] for i in range(len(c))}
        context = {
            "upper": concept,
            "incl_descr": incl_descr
        }
        return render(request, path, context)
    except Exception:
        return render(request, path, {})


def advise_he_form_view(request, *args, **kwargs):
    path = os.path.join('he_form.html')
    return render(request, path, {})


def advise_he_view(request, *args, **kwargs):
    preferred = request.POST.getlist('preferred')
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

    context = {profile: [(1 if profile in preferred else None)] for profile in profiles}
    context.update({"region": request.POST.get('region'),
                    "dormitory": request.POST.get('dormitory')})
    context_pd = pd.DataFrame(data=context)

    model = request.POST.get('model')
    result, accuracy = predict_he(context_pd, model)

    path = os.path.join('he_advise.html')
    return render(request, path, {"he": result[0], "accuracy": f"{accuracy * 100:.2f}"})


def predict_he(to_predict, preferred):
    adv = Advise()
    main(adv, preferred)
    result = advise_he(to_predict, adv)

    accuracy = accuracy_score(adv.Y, adv.model.predict(adv.X))

    return result, accuracy


def state_list_view(request, *args, **kwargs):
    path = os.path.join('states.html')
    st_date = """select s.name, fd.datevalue, rb.name
                from states as s
                inner join foundation_date as fd
                on s.id = fd.stateid
                left join official_religion as oreg
                on s.id = oreg.stateid
                left join religion_branches as rb
                on oreg.offrelid = rb.id;"""
    try:
        c = query(st_date)
        context = {
            "state_data": c
        }
        return render(request, path, context)
    except Exception:
        return render(request, path, {})


if __name__ == '__main__':
    pass

