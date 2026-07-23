
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# -----------------------------------
# CONFIGURACIÓN
# -----------------------------------
st.set_page_config(
    page_title="Sistema de Análisis de Felicidad",
    page_icon="😊",
    layout="wide"
)

st.title("😊 Sistema Inteligente de Análisis de Felicidad")
st.markdown("Simulación, análisis y visualización dinámica de datos de felicidad")

# -----------------------------------
# GENERACIÓN DE DATOS
# -----------------------------------
@st.cache_data
def generar_datos(n_personas):

    np.random.seed(42)

    edades = np.random.randint(18, 70, n_personas)

    ingresos = np.random.normal(
        3000,
        1200,
        n_personas
    ).clip(500, 10000)

    salud = np.random.randint(1, 11, n_personas)

    relaciones = np.random.randint(1, 11, n_personas)

    trabajo = np.random.randint(1, 11, n_personas)

    felicidad = (
        ingresos * 0.0015
        + salud * 2.5
        + relaciones * 3
        + trabajo * 2
        + np.random.normal(0, 5, n_personas)
    )

    felicidad = np.clip(felicidad, 0, 100)

    df = pd.DataFrame({
        "Edad": edades,
        "Ingresos": ingresos,
        "Salud": salud,
        "Relaciones": relaciones,
        "Trabajo": trabajo,
        "Felicidad": felicidad
    })

    return df


# -----------------------------------
# PANEL LATERAL
# -----------------------------------
st.sidebar.header("⚙️ Configuración")

cantidad = st.sidebar.slider(
    "Número de personas",
    100,
    5000,
    1000
)

df = generar_datos(cantidad)

edad_min = int(df["Edad"].min())
edad_max = int(df["Edad"].max())

rango_edad = st.sidebar.slider(
    "Rango de edad",
    edad_min,
    edad_max,
    (edad_min, edad_max)
)

df_filtrado = df[
    (df["Edad"] >= rango_edad[0]) &
    (df["Edad"] <= rango_edad[1])
]

# -----------------------------------
# CLASIFICACIÓN CUALITATIVA
# -----------------------------------
def clasificar(valor):

    if valor < 40:
        return "Baja"

    elif valor < 70:
        return "Media"

    return "Alta"

df_filtrado["Nivel_Felicidad"] = (
    df_filtrado["Felicidad"]
    .apply(clasificar)
)

# -----------------------------------
# MÉTRICAS
# -----------------------------------
st.header("📊 Resumen General")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Personas",
    len(df_filtrado)
)

c2.metric(
    "Felicidad Promedio",
    round(df_filtrado["Felicidad"].mean(), 2)
)

c3.metric(
    "Ingreso Promedio",
    f"${df_filtrado['Ingresos'].mean():,.0f}"
)

c4.metric(
    "Edad Promedio",
    round(df_filtrado["Edad"].mean(), 1)
)

# -----------------------------------
# ANALISIS CUANTITATIVO
# -----------------------------------
st.header("📈 Análisis Cuantitativo")

correlacion = df_filtrado.corr(numeric_only=True)

st.subheader("Matriz de Correlación")

fig_corr = px.imshow(
    correlacion,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Viridis"
)

st.plotly_chart(
    fig_corr,
    use_container_width=True
)

# -----------------------------------
# GRAFICO DISPERSION
# -----------------------------------
st.subheader("Relación Ingresos vs Felicidad")

fig_scatter = px.scatter(
    df_filtrado,
    x="Ingresos",
    y="Felicidad",
    color="Nivel_Felicidad",
    size="Salud",
    hover_data=["Edad"]
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)

# -----------------------------------
# ANALISIS CUALITATIVO
# -----------------------------------
st.header("📝 Análisis Cualitativo")

conteo = (
    df_filtrado["Nivel_Felicidad"]
    .value_counts()
    .reset_index()
)

conteo.columns = [
    "Nivel",
    "Cantidad"
]

fig_pie = px.pie(
    conteo,
    names="Nivel",
    values="Cantidad",
    title="Distribución de Niveles de Felicidad"
)

st.plotly_chart(
    fig_pie,
    use_container_width=True
)

porcentaje_alta = (
    (df_filtrado["Nivel_Felicidad"] == "Alta")
    .mean() * 100
)

porcentaje_media = (
    (df_filtrado["Nivel_Felicidad"] == "Media")
    .mean() * 100
)

porcentaje_baja = (
    (df_filtrado["Nivel_Felicidad"] == "Baja")
    .mean() * 100
)

st.markdown("### Interpretación")

if porcentaje_alta > 50:
    st.success(
        "La mayoría de las personas presenta niveles altos de felicidad."
    )

elif porcentaje_baja > 40:
    st.error(
        "Existe una proporción importante de personas con baja felicidad."
    )

else:
    st.warning(
        "La población presenta niveles intermedios de felicidad."
    )

# -----------------------------------
# SEGMENTACION
# -----------------------------------
st.header("🤖 Segmentación Inteligente")

variables = df_filtrado[
    [
        "Ingresos",
        "Salud",
        "Relaciones",
        "Trabajo",
        "Felicidad"
    ]
]

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df_filtrado["Cluster"] = (
    kmeans.fit_predict(variables)
)

fig_cluster = px.scatter(
    df_filtrado,
    x="Ingresos",
    y="Felicidad",
    color=df_filtrado["Cluster"].astype(str),
    title="Clusters de Personas"
)

st.plotly_chart(
    fig_cluster,
    use_container_width=True
)

# -----------------------------------
# ANALISIS DINAMICO
# -----------------------------------
st.header("🎯 Exploración Dinámica")

variable = st.selectbox(
    "Seleccione una variable",
    [
        "Ingresos",
        "Salud",
        "Relaciones",
        "Trabajo",
        "Felicidad"
    ]
)

fig_hist = px.histogram(
    df_filtrado,
    x=variable,
    nbins=25,
    title=f"Distribución de {variable}"
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)

# -----------------------------------
# DATOS
# -----------------------------------
st.header("📋 Datos Simulados")

st.dataframe(
    df_filtrado,
    use_container_width=True
)

# -----------------------------------
# CONCLUSIONES AUTOMATICAS
# -----------------------------------
st.header("📌 Conclusiones Automáticas")

factor_principal = (
    correlacion["Felicidad"]
    .drop("Felicidad")
    .abs()
    .idxmax()
)

st.info(
    f"El factor con mayor influencia sobre la felicidad es: {factor_principal}"
)

st.success(
    f"Nivel promedio de felicidad: {df_filtrado['Felicidad'].mean():.2f}"
)
