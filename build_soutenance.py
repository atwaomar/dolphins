"""Construit la soutenance (15 min) à partir du modèle « Ateliers d'apprentissage.pptx »."""
import copy
import os

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

W = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(W, 'soutenance.pptx')

TEAL = RGBColor(0x1A, 0x99, 0x88)
ORANGE = RGBColor(0xEB, 0x56, 0x00)
DARK = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xE9, 0xED, 0xEE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TEAL_LIGHT = RGBColor(0xD6, 0xEF, 0xEB)
HEAD, BODY = 'Raleway', 'Lato'

prs = Presentation(os.path.join(W, 'source.pptx'))
L = {l.name: l for l in prs.slide_layouts}

# on garde la page de titre et le sommaire, on retire les diapositives vides du modèle
sld_ids = prs.slides._sldIdLst
for sid in list(sld_ids)[2:]:
    prs.part.drop_rel(sid.rId)
    sld_ids.remove(sid)


# ---------------------------------------------------------------- outils
def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def title(slide, text, top=0.63):
    t = slide.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.8), Inches(top), Inches(8.41), Inches(0.59)
    t.text_frame.text = text
    return t


def box(slide, x, y, w, h, text='', size=14, bold=False, color=DARK, align=PP_ALIGN.LEFT,
        anchor=MSO_ANCHOR.TOP, font=BODY, name=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        runs = line if isinstance(line, list) else [(line, {})]
        for rt, opt in runs:
            r = p.add_run()
            r.text = rt
            r.font.name = opt.get('font', font)
            r.font.size = Pt(opt.get('size', size))
            r.font.bold = opt.get('bold', bold)
            r.font.color.rgb = opt.get('color', color)
            r.font.italic = opt.get('italic', False)
        p.alignment = align
        p.space_after = Pt(opt.get('after', 6)) if runs else Pt(6)
    if name:
        tb.name = name
    return tb


def card(slide, x, y, w, h, fill=LIGHT, name=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.08
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


def circle(slide, x, y, d, text, fill=TEAL, size=16):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = HEAD, Pt(size), True, WHITE
    return s


def picture(slide, fname, x, y, w=None, h=None):
    im = Image.open(os.path.join(W, fname))
    ar = im.size[0] / im.size[1]
    if w is not None and h is not None:          # tenir dans la boîte en gardant les proportions
        if w / h > ar:
            nw = h * ar
            x += (w - nw) / 2
            w = nw
        else:
            nh = w / ar
            y += (h - nh) / 2
            h = nh
    elif w is not None:
        h = w / ar
    else:
        w = h * ar
    return slide.shapes.add_picture(os.path.join(W, fname), Inches(x), Inches(y), Inches(w), Inches(h))


def arrow(slide, x, y, w, h, fill=GREY):
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def style_chart(chart, size=11):
    chart.has_title = False
    chart.font.name = BODY
    chart.font.size = Pt(size)
    chart.font.color.rgb = GREY


def fr_labels(plot, values, color=DARK, size=12):
    """Étiquettes écrites à la française (virgule décimale), une par barre."""
    for pt, v in zip(plot.series[0].points, values):
        dl = pt.data_label
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        tf = dl.text_frame
        tf.text = f'{v:.1f}'.replace('.', ',') + ' %'
        r = tf.paragraphs[0].runs[0]
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), True, color, BODY


def new(layout='TITLE_ONLY'):
    return prs.slides.add_slide(L[layout])


# ---------------------------------------------------------------- 1. titre
s = prs.slides[0]
s.shapes.title.text_frame.text = "Classification de clics d'odontocètes"
sub = s.placeholders[1].text_frame
sub.text = ("Tom VILLEMAGNE, Benjamin ROYER, Nicolas HUYGHE, Omar ATWA — 3A SDIA-ISSD, "
            "Télécom Physique Strasbourg\nChallenge Data DOCC10 — Ateliers d'apprentissage automatique")
notes(s, "Bonjour. Notre projet porte sur le challenge DOCC10 : reconnaître automatiquement l'espèce "
         "d'un dauphin ou d'une baleine à dents à partir d'un seul de ses clics d'écholocation. "
         "Nous allons voir qu'un très bon score peut cacher un biais, et comment nous l'avons mis en "
         "évidence puis corrigé. (≈ 20 s)")

# ---------------------------------------------------------------- 2. plan
s = prs.slides[1]
s.shapes.title.text_frame.text = 'Plan'
body = s.placeholders[1]
tf = body.text_frame
first_p = copy.deepcopy(tf.paragraphs[0]._p)
for p in list(tf.paragraphs)[1:]:
    p._p.getparent().remove(p._p)
items = ['Contexte et objectif', 'Données et prétraitement', 'Trois modèles',
         "Un biais d'enregistrement", 'Correction et résultats au challenge', 'Conclusion']
tf.paragraphs[0].runs[0].text = items[0]
for it in items[1:]:
    np_ = copy.deepcopy(first_p)
    tf._txBody.append(np_)
    tf.paragraphs[-1].runs[0].text = it
for p in tf.paragraphs:
    for r in p.runs:
        r.font.size = Pt(18)
        r.font.bold = False
    p.space_after = Pt(10)
notes(s, "Nous présentons d'abord le contexte et les données, puis le prétraitement et trois "
         "modèles. Le cœur de la présentation est la mise en évidence d'un biais d'enregistrement, "
         "puis sa correction et nos résultats au challenge. (≈ 20 s)")

# ---------------------------------------------------------------- 3. contexte
s = new('TITLE_ONLY')
title(s, 'Contexte : écouter les odontocètes')
box(s, 0.8, 1.6, 4.6, 3.4, [
    [('Challenge Data ', {'bold': True}), ("proposé par l'ADSIL et l'Université de Toulon", {})],
    [('20 stations ', {'bold': True}), ("d'enregistrement sous-marin dans les Caraïbes", {})],
    [("Les odontocètes émettent des ", {}), ('clics', {'bold': True}),
     (" pour s'orienter et chasser (écholocation)", {})],
    [('Objectif : ', {'bold': True, 'color': TEAL}),
     ("reconnaître l'espèce à partir d'un seul clic, parmi ", {}), ('10 espèces', {'bold': True})],
], size=16)
picture(s, 'dauphin.jpg', 5.7, 1.6, w=3.5, h=3.0)
box(s, 5.7, 4.65, 3.5, 0.3, 'Source : pratique.fr', size=10, color=GREY, align=PP_ALIGN.RIGHT)
notes(s, "Le challenge est proposé par l'ADSIL, qui exploite une vingtaine de stations "
         "d'enregistrement dans les Caraïbes. Les odontocètes, c'est-à-dire les cétacés à dents, "
         "émettent des clics pour s'orienter et chasser. Analyser ces clics permet de savoir quelles "
         "espèces passent près des stations. Notre objectif : à partir d'un seul clic, retrouver "
         "l'espèce parmi dix classes. (≈ 1 min)")

# ---------------------------------------------------------------- 4. données
s = new('TITLE_ONLY')
title(s, 'Les données DOCC10')
stats = [('113 120', 'fenêtres étiquetées'), ('10', 'espèces équilibrées'),
         ('41 ms', '8192 échantillons à 200 kHz'), ('Accuracy', 'métrique du challenge')]
for i, (big, small) in enumerate(stats):
    x = 0.8 + i * 2.15
    card(s, x, 1.55, 1.95, 1.05)
    box(s, x, 1.62, 1.95, 0.5, big, size=24, bold=True, color=TEAL, align=PP_ALIGN.CENTER, font=HEAD)
    box(s, x + 0.1, 2.12, 1.75, 0.45, small, size=11, color=GREY, align=PP_ALIGN.CENTER)
picture(s, 'signaux_par_espece.png', 0.8, 2.8, w=8.4, h=2.3)
notes(s, "Le jeu d'entraînement compte 113 120 fenêtres, parfaitement équilibrées entre les dix "
         "espèces. Chaque fenêtre dure 41 millisecondes, soit 8192 échantillons à 200 kHz, avec "
         "le clic au centre. Le jeu de test, 20 960 fenêtres, n'est pas étiqueté : il ne sert qu'à la "
         "soumission. La métrique imposée est l'accuracy. On voit en bas une fenêtre par espèce : "
         "le clic est très bref, et c'est le bruit de fond qui domine le signal. (≈ 1 min)")

# ---------------------------------------------------------------- 5. prétraitement
s = new('TITLE_ONLY')
title(s, 'Prétraitement : isoler le clic')
steps = [('Filtre passe-haut à 5 kHz', "95 à 100 % de l'énergie du bruit est sous 5 kHz"),
         ('Recentrage', "fenêtre de 1024 échantillons autour du maximum de l'enveloppe"),
         ('Normalisation', 'les amplitudes varient de 27 dB selon la distance')]
for i, (h, d) in enumerate(steps):
    y = 1.65 + i * 1.12
    circle(s, 0.8, y, 0.55, str(i + 1))
    box(s, 1.55, y - 0.02, 3.5, 0.35, h, size=16, bold=True, font=HEAD)
    box(s, 1.55, y + 0.36, 3.5, 0.6, d, size=13, color=GREY)
picture(s, 'coupure_spectres.png', 5.3, 1.55, w=3.9, h=3.0)
box(s, 5.3, 4.65, 3.9, 0.3, 'Spectre moyen du clic et du bruit', size=10, color=GREY, align=PP_ALIGN.CENTER)
notes(s, "Le prétraitement isole le clic en trois étapes. D'abord un filtre passe-haut à 5 kHz : "
         "le graphique montre que sous 5 kHz, le clic et le bruit se confondent, et cette bande "
         "contient presque toute l'énergie du bruit. Ensuite, on recentre une fenêtre de 1024 "
         "échantillons sur le maximum de l'enveloppe du signal, car le clic du cachalot n'est pas "
         "toujours au centre. Enfin, on normalise l'amplitude, qui dépend surtout de la distance de "
         "l'animal. (≈ 1 min)")

# ---------------------------------------------------------------- 6. trois modèles
s = new('TITLE_ONLY')
title(s, 'Trois modèles')
models = [('Ridge', 'Régression linéaire régularisée, modèle de référence du challenge'),
          ('Gradient boosting', 'Arbres de décision sur le log-spectre du clic (domaine fréquentiel)'),
          ('CNN 1D', 'Réseau convolutif sur le signal temporel, 5 blocs convolutifs')]
for i, (h, d) in enumerate(models):
    x = 0.8 + i * 2.85
    card(s, x, 1.55, 2.65, 1.25, fill=TEAL_LIGHT if i == 2 else LIGHT)
    box(s, x + 0.15, 1.65, 2.35, 0.35, h, size=16, bold=True, color=TEAL if i == 2 else DARK, font=HEAD)
    box(s, x + 0.15, 2.02, 2.35, 0.75, d, size=12, color=GREY)
picture(s, 'cnn1d.png', 0.8, 3.0, w=8.4, h=2.05)
notes(s, "Nous avons comparé trois familles de modèles. Le Ridge, une régression linéaire, est la "
         "référence fournie par le challenge. Le gradient boosting travaille dans le domaine "
         "fréquentiel, sur le spectre du clic. Enfin, un réseau convolutif 1D travaille directement "
         "sur le signal : cinq blocs de convolutions, puis un résumé par la moyenne et le maximum, et "
         "une couche dense qui donne l'espèce. (≈ 1 min)")

# ---------------------------------------------------------------- 7. résultats modèles
s = new('TITLE_ONLY')
title(s, 'Le CNN domine nettement')
cd = CategoryChartData()
cd.categories = ['Ridge (signal)', 'Ridge (log-spectre)', 'Gradient boosting', 'CNN 1D']
cd.add_series('Accuracy (%)', (18.0, 68.4, 82.2, 93.9))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.8), Inches(1.5), Inches(5.4), Inches(3.6), cd)
ch = gf.chart
style_chart(ch)
ch.has_legend = False
pl = ch.plots[0]
pl.gap_width = 60
fr_labels(pl, (18.0, 68.4, 82.2, 93.9))
ser = pl.series[0]
for i, pt in enumerate(ser.points):
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = TEAL if i == 3 else RGBColor(0xA9, 0xB7, 0xBB)
va = ch.value_axis
va.minimum_scale, va.maximum_scale = 0, 112
va.has_major_gridlines = False
va.visible = False
ch.category_axis.tick_labels.font.size = Pt(12)
ch.category_axis.format.line.fill.background()
card(s, 6.5, 1.6, 2.7, 3.3)
box(s, 6.7, 1.75, 2.3, 3.1, [
    [('+12 points', {'bold': True, 'size': 20, 'color': TEAL, 'font': HEAD})],
    [("pour le CNN sur le gradient boosting, à entrée égale", {'size': 13})],
    [('± 0,1 à 0,8 point', {'bold': True, 'size': 16, 'color': DARK, 'font': HEAD})],
    [('validation croisée à 5 plis : les écarts sont nets', {'size': 13})],
])
box(s, 0.8, 5.12, 5.4, 0.3, 'Fenêtre prétraitée de 1024 échantillons, découpage aléatoire', size=10, color=GREY)
notes(s, "Voici les résultats en validation croisée à cinq plis, sur la même fenêtre de 1024 "
         "échantillons pour tous. Le Ridge sur le signal est presque au hasard : un clic et son "
         "opposé ont la même espèce, ce qu'un modèle linéaire ne peut pas représenter. Sur le "
         "spectre, il monte à 68 %, le gradient boosting à 82 %, et le CNN atteint 94 %, soit 12 "
         "points de plus. Les écarts-types sont inférieurs à un point : la différence est nette. "
         "(≈ 1 min)")

# ---------------------------------------------------------------- 8. ablation
s = new('TITLE_ONLY')
title(s, 'Un indice : moins on nettoie, mieux ça marche')
cd = CategoryChartData()
cd.categories = ['Prétraitement complet', 'Sans recentrage', 'Sans normalisation',
                 'Sans prétraitement', 'Signal brut complet']
cd.add_series('Accuracy (%)', (93.8, 94.1, 94.5, 95.1, 97.1))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.8), Inches(1.5), Inches(5.6), Inches(3.6), cd)
ch = gf.chart
style_chart(ch)
ch.has_legend = False
pl = ch.plots[0]
pl.gap_width = 50
fr_labels(pl, (93.8, 94.1, 94.5, 95.1, 97.1))
for i, pt in enumerate(pl.series[0].points):
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = ORANGE if i == 4 else RGBColor(0xA9, 0xB7, 0xBB)
va = ch.value_axis
va.minimum_scale, va.maximum_scale = 90, 98
va.has_major_gridlines = False
va.visible = False
ch.category_axis.tick_labels.font.size = Pt(11)
ch.category_axis.format.line.fill.background()
card(s, 6.7, 1.6, 2.5, 3.3)
box(s, 6.9, 1.75, 2.1, 3.1, [
    [('97,1 %', {'bold': True, 'size': 26, 'color': ORANGE, 'font': HEAD})],
    [('sur le signal brut, qui contient surtout du bruit de fond', {'size': 13})],
    [('Question', {'bold': True, 'size': 15, 'color': DARK, 'font': HEAD})],
    [("le bruit de fond aide-t-il à reconnaître l'espèce ?", {'size': 13})],
])
notes(s, "Un résultat nous a intrigués. En retirant les étapes de prétraitement une par une, "
         "l'accuracy augmente : 94,1 % sans recentrage, 95,1 % sans aucun prétraitement, et 97,1 % "
         "sur le signal brut complet. Or ce signal brut est surtout du bruit de fond. Les écarts sont "
         "bien supérieurs à la variabilité entre plis. Si le bruit aide à reconnaître l'espèce, c'est "
         "qu'il contient une information propre à l'enregistrement. (≈ 1 min)")

# ---------------------------------------------------------------- 9. section
s = new('TITLE')
s.shapes.title.text_frame.text = "L'animal… ou l'enregistrement ?"
s.placeholders[1].text_frame.text = "Mise en évidence d'un biais d'enregistrement"
notes(s, "C'est la question centrale de notre travail : le modèle reconnaît-il l'animal, ou "
         "reconnaît-il l'enregistrement dont provient le signal ? (≈ 10 s)")

# ---------------------------------------------------------------- 10. bruit seul
s = new('TITLE_ONLY')
title(s, 'Le bruit seul suffit à deviner l\'espèce')
card(s, 0.8, 1.6, 2.6, 2.0)
box(s, 0.8, 1.75, 2.6, 0.75, '74,6 %', size=36, bold=True, color=ORANGE, align=PP_ALIGN.CENTER, font=HEAD)
box(s, 0.95, 2.55, 2.3, 0.9, 'CNN sur le bruit seul, sans aucun clic (découpage aléatoire)',
    size=12, color=GREY, align=PP_ALIGN.CENTER)
arrow(s, 3.55, 2.4, 0.6, 0.4)
card(s, 4.3, 1.6, 2.6, 2.0)
box(s, 4.3, 1.75, 2.6, 0.75, '49,3 %', size=36, bold=True, color=TEAL, align=PP_ALIGN.CENTER, font=HEAD)
box(s, 4.45, 2.55, 2.3, 0.9, 'en validant sur des enregistrements jamais vus', size=12, color=GREY,
    align=PP_ALIGN.CENTER)
box(s, 7.15, 1.75, 2.1, 1.8, [
    [('Hasard : 10 %', {'bold': True, 'size': 16, 'font': HEAD})],
    [('Le bruit de fond porte une information sur la station', {'size': 13, 'color': GREY})],
])
box(s, 0.8, 3.85, 8.4, 1.2, [
    [('Pseudo-sessions : ', {'bold': True, 'color': TEAL}),
     ("le challenge ne dit pas de quel enregistrement vient chaque fenêtre. On regroupe donc les "
      "fenêtres selon le spectre de leur bruit de fond (k-moyennes), et des groupes entiers "
      "partent en validation.", {})],
], size=14)
notes(s, "Pour le vérifier, on entraîne le même réseau sur les dix premières millisecondes de "
         "chaque fenêtre, loin du clic : il ne voit que du bruit. Il devine pourtant l'espèce dans "
         "75 % des cas. Problème : avec un découpage aléatoire, des fenêtres du même enregistrement "
         "sont à la fois en entraînement et en validation. Comme le challenge ne fournit pas "
         "l'enregistrement d'origine, nous avons reconstruit des pseudo-sessions en regroupant les "
         "fenêtres par spectre de bruit. En validant sur des groupes jamais vus, le score tombe à "
         "49 %, mais reste cinq fois au-dessus du hasard. (≈ 1 min 15)")

# ---------------------------------------------------------------- 11. test d'échange
s = new('TITLE_ONLY')
title(s, "Le test d'échange du bruit")
# schéma de la chimère
def strip(slide, y, label, noise_txt, click_txt, noise_fill, click_fill):
    box(slide, 0.8, y + 0.08, 1.2, 0.35, label, size=13, bold=True, font=HEAD)
    for j, (x, w, txt, fill) in enumerate([(2.05, 1.5, noise_txt, noise_fill), (3.6, 1.0, click_txt, click_fill),
                                           (4.65, 1.5, noise_txt, noise_fill)]):
        r = card(slide, x, y, w, 0.5, fill=fill)
        tf = r.text_frame
        tf.margin_left = tf.margin_right = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = txt
        run.font.name, run.font.size, run.font.bold = BODY, Pt(12), True
        run.font.color.rgb = WHITE if fill in (TEAL, ORANGE) else DARK


strip(s, 1.6, 'Fenêtre A', 'bruit GG', 'clic GG', LIGHT, TEAL)
strip(s, 2.25, 'Fenêtre B', 'bruit ZC', 'clic ZC', RGBColor(0xFB, 0xDD, 0xCC), ORANGE)
strip(s, 3.15, 'Chimère', 'bruit ZC', 'clic GG', RGBColor(0xFB, 0xDD, 0xCC), TEAL)
box(s, 0.8, 3.85, 5.4, 1.2, [
    [('Réponse GG : ', {'bold': True, 'color': TEAL}), ('le modèle suit le clic', {})],
    [('Réponse ZC : ', {'bold': True, 'color': ORANGE}), ('le modèle suit le bruit', {})],
], size=14)
card(s, 6.55, 1.6, 2.65, 3.3)
box(s, 6.75, 1.72, 2.25, 3.1, [
    [('57,9 %', {'bold': True, 'size': 26, 'color': TEAL, 'font': HEAD})],
    [('des chimères classées selon le clic', {'size': 12})],
    [('25,1 %', {'bold': True, 'size': 26, 'color': ORANGE, 'font': HEAD})],
    [('selon le bruit', {'size': 12})],
    [('Cachalot : 3 %', {'bold': True, 'size': 15, 'font': HEAD})],
    [('de clics reconnus dans un autre bruit', {'size': 12})],
])
notes(s, "Pour savoir si le modèle utilise vraiment le bruit, on lui pose une question piège : on "
         "prend le clic d'un dauphin de Risso et on le colle dans le bruit de fond d'une baleine à bec "
         "de Cuvier, au même niveau sonore. S'il répond GG, il suit le clic ; s'il répond ZC, il suit "
         "le bruit. Le CNN sur le signal brut ne suit le clic que dans 58 % des cas, et suit le bruit "
         "dans un quart des cas. Pour le cachalot c'est flagrant : son clic placé dans un autre bruit "
         "n'est reconnu que dans 3 % des cas. Le modèle reconnaît donc en partie l'enregistrement. "
         "(≈ 1 min 15)")

# ---------------------------------------------------------------- 12. correction
s = new('TITLE_ONLY')
title(s, "Correction : échanger le bruit à l'entraînement")
box(s, 0.8, 1.55, 3.4, 3.5, [
    [("Pendant l'entraînement, avec une probabilité ", {}), ('p', {'italic': True, 'bold': True}),
     (", on garde le clic et on remplace le bruit par celui d'une autre fenêtre.", {})],
    [("Le bruit ne prédit plus l'espèce : le réseau doit apprendre le clic.", {'bold': True, 'color': TEAL})],
    [('p = 0,5 : suivi du clic 57,9 → 88,8 %, pour seulement −0,7 point d\'accuracy.', {})],
], size=14)
cd = CategoryChartData()
cd.categories = ['0', '0,25', '0,5', '0,75', '1']
cd.add_series('Suit le clic (%)', (57.9, 84.7, 88.8, 89.9, 89.6))
cd.add_series('Accuracy (%)', (93.5, 93.4, 92.8, 92.6, 91.7))
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(4.4), Inches(1.5), Inches(4.85), Inches(3.4), cd)
ch = gf.chart
style_chart(ch)
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(11)
for ser, col in zip(ch.plots[0].series, (TEAL, ORANGE)):
    ser.format.line.color.rgb = col
    ser.format.line.width = Pt(2.5)
    ser.smooth = False
    ser.marker.format.fill.solid()
    ser.marker.format.fill.fore_color.rgb = col
    ser.marker.format.line.color.rgb = col
va = ch.value_axis
va.minimum_scale, va.maximum_scale = 50, 100
va.major_gridlines.format.line.color.rgb = LIGHT
va.format.line.fill.background()
va.tick_labels.font.size = Pt(10)
ch.category_axis.tick_labels.font.size = Pt(11)
ch.category_axis.has_title = True
ch.category_axis.axis_title.text_frame.text = "Probabilité d'échange p"
ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
box(s, 4.4, 5.0, 4.85, 0.3, 'Validation par pseudo-sessions, signal brut complet', size=10, color=GREY)
notes(s, "La correction réutilise la même opération, mais pendant l'entraînement. Avec une "
         "probabilité p, on garde le clic d'une fenêtre et on remplace son bruit par celui d'une "
         "autre fenêtre du batch. Le bruit ne permet alors plus de prédire l'espèce, le réseau est "
         "obligé d'apprendre le clic. Le graphique montre l'effet de p : le suivi du clic bondit dès "
         "p = 0,25, puis plafonne vers 89 %, tandis que l'accuracy baisse doucement. Nous retenons "
         "p = 0,5 : le suivi du clic passe de 58 à 89 %, pour moins d'un point d'accuracy perdu. "
         "Le cachalot passe de 3 à 98 %. (≈ 1 min 15)")

# ---------------------------------------------------------------- 13. challenge
s = new('TITLE_ONLY')
title(s, 'Résultat au challenge')
card(s, 0.8, 1.6, 2.5, 1.75)
box(s, 0.8, 1.7, 2.5, 0.75, '80,8 %', size=36, bold=True, color=TEAL, align=PP_ALIGN.CENTER, font=HEAD)
box(s, 0.95, 2.5, 2.2, 0.8, 'sur le jeu de test officiel, quatrième place', size=13, color=GREY,
    align=PP_ALIGN.CENTER)
picture(s, 'challenge-logo.png', 3.6, 1.6, w=2.4, h=0.55)
picture(s, 'challenge.png', 3.6, 2.2, w=5.6, h=1.3)
box(s, 0.8, 3.75, 8.4, 1.4, [
    [('Même score sans correction : ', {'bold': True}),
     ("l'échange de bruit rend le modèle plus robuste, sans améliorer ce score.", {})],
    [('Écart avec la validation (−12 points) : ', {'bold': True}),
     ("un classifieur distingue le bruit du test de celui du train (AUC 0,92). Le test vient "
      "probablement d'enregistrements jamais vus.", {})],
], size=14)
notes(s, "Sur le jeu de test du challenge, notre modèle obtient 80,8 % et se classe quatrième. "
         "Deux remarques honnêtes. D'abord, le modèle sans correction obtient le même score : la "
         "correction rend le modèle plus robuste au bruit, mais ne change pas ce score. Ensuite, on "
         "perd 12 points par rapport à la validation. Pour comprendre, nous avons entraîné un "
         "classifieur à distinguer une fenêtre du train d'une fenêtre du test à partir de son seul "
         "bruit : il y arrive très bien, avec une AUC de 0,92. Le test semble donc venir "
         "d'enregistrements que nous n'avons jamais vus. (≈ 1 min 15)")

# ---------------------------------------------------------------- 14. conclusion
s = new('TITLE_ONLY')
title(s, 'Conclusion')
pts = [('93,9 %', 'Le CNN 1D sur le signal dépasse nettement les modèles fréquentiels'),
       ('Biais', "Une très bonne accuracy peut venir du bruit de fond : il faut vérifier sur quoi "
                 "le modèle s'appuie"),
       ('88,8 %', "Avec l'échange de bruit, le modèle suit le clic, pour une accuracy presque inchangée")]
for i, (big, txt) in enumerate(pts):
    x = 0.8 + i * 2.85
    card(s, x, 1.55, 2.65, 2.2, fill=TEAL_LIGHT if i == 1 else LIGHT)
    box(s, x, 1.7, 2.65, 0.6, big, size=26, bold=True, color=ORANGE if i == 1 else TEAL,
        align=PP_ALIGN.CENTER, font=HEAD)
    box(s, x + 0.2, 2.35, 2.25, 1.3, txt, size=13, color=DARK, align=PP_ALIGN.CENTER)
box(s, 0.8, 4.0, 8.4, 1.0, [
    [('Pistes : ', {'bold': True, 'color': TEAL}),
     ("obtenir l'enregistrement d'origine de chaque fenêtre, et tirer le bruit de remplacement dans "
      "une autre pseudo-session.", {})],
], size=14)
notes(s, "Pour conclure, trois messages. Le CNN 1D sur le signal temporel dépasse nettement les "
         "modèles fréquentiels. Surtout, une très bonne accuracy peut cacher un biais : ici, le "
         "modèle reconnaissait en partie l'enregistrement. Il faut donc vérifier sur quoi un modèle "
         "s'appuie, pas seulement son score. Enfin, l'échange de bruit corrige ce comportement. Pour "
         "aller plus loin, il faudrait connaître l'enregistrement d'origine de chaque fenêtre, ou "
         "tirer le bruit de remplacement dans une autre pseudo-session. (≈ 1 min)")

# ---------------------------------------------------------------- 15. merci
s = new('TITLE')
s.shapes.title.text_frame.text = 'Merci de votre attention'
s.placeholders[1].text_frame.text = 'Questions ?'
notes(s, "Merci de votre attention, nous sommes prêts à répondre à vos questions.")

prs.save(OUT)
print('écrit :', OUT, len(prs.slides), 'diapositives')
