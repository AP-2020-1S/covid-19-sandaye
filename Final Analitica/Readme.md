## URL Dashboard ##
## https://sangonzalez.github.io/


# Pronóstico de la evolución de casos confirmados,recuperados y muertes de SARS-CoV-2 para las 5 principales ciudades de Colombia

### Equipo de trabajo:

* _Santiago Gonzalez Cardona_

* _Dairo Alberto Cuervo Garcia_

* _Yeison Alexander Florez Calderon_


# DEFINICIÓN DEL PROBLEMA

Con la presencia del Coronavirus (Covid-19) en el territorio nacional y los riesgos que eso supone para la salud de los colombianos, fue necesario para el Gobierno Nacional adoptar disposiciones normativas de emergencia que propendan por la prevención, protección y garantía en el acceso a servicios de salud de los mismos.Decisiones que evidentemente han dejado daños profundos en la estructura económica del país.

Para el segundo semestre del 2020 el  PIB nacional cayó casi un 15%. Esta contracción económica ha sido la más profunda en la historia del país y ha dejado por delante un aumento significativo de las tasas de desempleo y  un deterioro en la calidad de vida de los colombianos.

Hasta finales de agosto del 2020 habían más de 650.000 casos confirmados , 498.000 recuperados y  aproximadamente 20.000 muertes.Adicionalmente ,la incertidumbre asociada con el comportamiento del virus en el mediano y largo plazo es uno de los principales problemas de interés.Información que es de vital importancia para tomar decisiones efectivas a nivel gubernamental como por ejemplo , temas de infraestructura , políticas de confinamiento  y decisiones relacionadas con el comercio internacional  y/o  apertura económica.


# DEFINICIÓN DEL PROBLEMA DE ANALÍTICA

Para la toma de decisiones efectivas es necesario que el producto final haya sido construido o trazado bajo metodologías estructuradas , modelos precisos , confiables y debidamente soportados  donde  la evaluación de los  modelos sea cíclica y permita mejorar  constantemente  el resultado final. 

Teniendo en cuenta lo anterior , las pregunta que nos llevará a trazar la metodología del proyecto son: 
¿ Que modelo de pronóstico es ideal para proyectar información confiable en el corto o mediano plazo de los casos confirmados del  SARS-CoV-2 con sus respectivos desenlaces(recuperados o fallecidos)  para  las 5 principales ciudades de Colombia?
¿Qué metodología es la adecuada para permitir que el producto de datos sea autosuficiente , confiable y duradero?

En la literatura existen diferentes técnicas de pronóstico para las series de tiempo como por ejemplo: regresiones lineales , modelos autorregresivos y de medias móviles (ARIMA) , redes neuronales , modelo SIR etc. Estas técnicas poseen características especiales y unos con mejores ajustes que otros según el problema que sea planteado.Adicionalmente existen metodologías para la construcción de proyectos como la CRISP-DM , metodología que permite que el producto tenga unas fases de construcción definidas obteniendo así un mejor resultado.

## OBJETIVO
El objetivo de este proyecto es la predicción de corto y mediano plazo del total de casos confirmados, los nuevos casos, los casos activos, recuperados y muertes para las 5 principales ciudades de Colombia, utilizando técnicas estadísticas, de inteligencia artificial o modelos híbridos.

# DATOS
La fuente principal de información proviene de la página oficial del gobierno nacional datos.gov.co la cual contiene información actualizada dia a dia de los casos confirmados del  SARS-CoV-2 en el territorio nacional.

## Estructura de los datos

|  Variable             |    Tipo de dato         |  
|-----------------------|-------------------------|
|ID CASO                |  NUMÉRICO               |
|FECHA DE NOTIFICACIÓN  |  FECHA                  |
|CÓDIGO DIVIPOLA        |  NUMÉRICO               |
|CIUDAD DE UBICACIÓN    |  CATEGÓRICA             |
|DEPARTAMENTO O DISTRITO|  CATEGÓRICA             |
|ATENCIÓN               |  CATEGÓRICA (RECUPERADO , FALLECIDO , CASA , HOSPITAL , HOSPITAL UCI )                  |
|EDAD                   |  NUMÉRICO               |
|SEXO                   |  CATEGÓRICA (F.M)       |
|TIPO                   |  CATEGÓRICA (IMPORTADO , RELACIONADO , EN ESTUDIO)               |
|ESTADO                 |  CATEGÓRICA (ASINTOMÁTICO ,LEVE , MODERADO , GRAVE , FALLECIDO)       |
|PAÍS PROCEDENCIA       |  CATEGÓRICA             |
|FECHA DE MUERTE        |  FECHA                  |
|FECHA DIAGNÓSTICO      |  FECHA                  |
|FECHA RECUPERADO       |  FECHA                  |
|FECHA REPORTADO WEB    |  FECHA                  |
|CÓDIGO DEPARTAMENTO    |  NUMÉRICO               |
|CÓDIGO PAÍS            |  NUMÉRICO               |
|PERTENENCIA ÉTNICA     |  CATEGÓRICA             |



## Obtención y limpieza de datos

Con ayuda de un algoritmo de programación (URL indexing) los datos se descargan directamente de la fuente principal para ser almacenados y usados por el producto de datos.

Para la limpieza de los datos erróneos o faltantes se usaron diferentes técnicas de limpieza y homogeneización con el fin de arrojar una base de datos sólida y lista para hacer el trabajo de análisis.  Algunas de las técnicas usadas son la siguientes:

Homogeneización de caracteres 
Selección de datos relevantes  
Eliminación de datos innecesarios 
Cambios de tipo de datos

## Análisis Exploratorio de los datos




# 1. DESPLIEGUE DEL MODELO
## 1.2 Modelo ARIMA
Para el despliegue del modelo ARIMA se presentan los pronósticos a corto plazo del total de casos confirmados, nuevos casos, casos activos, recuperados y fallecidos para las 5 principales ciudades de colombia, para observar el comportamiento futuro del virus Covid-19 en las personas pertenecientes a las mismas.

## MEDELLÍN

A continuación se presenta el modelado y pronóstico para la ciudad de Medellín:

```python
import pandas as pd
import numpy as np
import datetime
import os
import os.path
import sklearn
! pip install pmdarima
from pmdarima.arima import auto_arima
from pmdarima.arima import ADFTest
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
!pip install plotly==4.9.0
import plotly.graph_objects as go
from plotly.offline import plot
```
```python
# Datos Recuperados Acumulados Medellín
df1 = pd.read_csv(os.path.join('../Output', 'data_medellin.csv'))
df1 = df1[['FECHA','RECUPERADOS_ACUMM']]
df3 = pd.to_datetime(df1['FECHA'])
df1.index = df3
data_recuperados = df1.drop(['FECHA'], axis=1)
```
```python
# Datos Muertes diarias Medellín
df4 = pd.read_csv(os.path.join('../Output', 'data_medellin.csv'))
df4 = df4[['FECHA','FALLECIDOS']]
df5 = pd.to_datetime(df4['FECHA'])
df4.index = df5
data_fallecidos_diarios = df4.drop(['FECHA'], axis=1)
```
```python
# Datos Muertes Acumuladas Medellín
df4 = pd.read_csv(os.path.join('../Output', 'data_medellin.csv'))
df4 = df4[['FECHA','FALLECIDOS_ACUMM']]
df5 = pd.to_datetime(df4['FECHA'])
df4.index = df5
data_fallecidos = df4.drop(['FECHA'], axis=1)
```
```python
# Datos Activos Medellín
df5 = pd.read_csv(os.path.join('../Output', 'data_medellin.csv'))
df5 = df5[['FECHA','ACTIVOS']]
df6 = pd.to_datetime(df5['FECHA'])
df5.index = df6
data_activos = df5.drop(['FECHA'], axis=1)
```
```python
# Datos Confirmados Medellín
df7 = pd.read_csv(os.path.join('../Output', 'data_medellin.csv'))
df7 = df7[['FECHA','CASOS_ACUMM']]
df8 = pd.to_datetime(df7['FECHA'])
df7.index = df8
data_confirmados = df7.drop(['FECHA'], axis=1)
```

### Construcción del modelo AUTO-ARIMA para los nuevos casos
```python
# Data, data_train y data_test
real = data_nuevoscasos
train = data_nuevoscasos[:int(len(data_nuevoscasos)*(1-7/len(data_nuevoscasos)))]
test = data_nuevoscasos[-7:]

modelo_arima = auto_arima(train, start_p=1, d=None, start_q=1, 
                          max_p=12, max_d=8, max_q=12,
                          start_P=1, D=None, start_Q=1, 
                          max_P=12, max_D=8, max_Q=12, max_order=None, m=7, seasonal=True,
                          trace=True, supress_warnings=True, stepwise=True, random_state=20, n_fits=50)
```
### Selección del mejor modelo:
```python
modelo_arima.summary()
```


SARIMAX Results
|Dep. Variable:|	y|	No. Observations:|	45|
|--------------|-----|-------------------|----|
|Model:|	SARIMAX|	Log Likelihood|	-144.911|
|Date:|	Sun, 06 Sep 2020|	AIC|	293.822|
|Time:|	01:07:35|	BIC|	297.436|
|Sample:|	0	|HQIC|	295.169|
|       |- 45|		
|Covariance Type:|	opg|		
||coef|std err|z| P>abs(z) | [0.025	0.975]|
|intercept|	19.5111|	0.904|	21.575	0.000	17.739	21.284
|sigma2| 36.6943	6.532	5.618	0.000	23.892	49.497
|Ljung-Box (Q):|	43.94	Jarque-Bera (JB):	1.30
|Prob(Q):|	0.31	Prob(JB):	0.52


![casos nuevos](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/fig/med_corto_casosnuevos.PNG)


```python
#test predicción con Auto Arima
test_prediccion_nuevoscasos = pd.DataFrame(modelo_arima.predict(n_periods = len(test)), index = test.index)
test_prediccion_nuevoscasos.columns = ['Test predicción nuevos casos']
```
```python
# Grafica de predicción
fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train['CASOS'],mode='lines',name='Train_Casos_Diarios'))
fig.add_trace(go.Scatter(x=test.index, y=test['CASOS'], mode='lines',name='Test_Casos_Diarios'))
fig.add_trace(go.Scatter(x=test_prediccion_nuevoscasos.index, y=test_prediccion_nuevoscasos['Test predicción nuevos casos'], mode='lines',name='Prediccion_Casos_Diarios'))
fig.update_layout(
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color='#3F3F3F',
        ),
        legend=dict(
            x=0.02,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color= '#474747',
            ),
            bgcolor='#FFFFFF',
            borderwidth=3
        ),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#F4EEEF',
        margin=dict(l=0, 
                    r=0, 
                    t=0, 
                    b=0
                    ),
      

    )
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#D9D8D8')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#D9D8D8')

fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#474747')
fig.show()
```
![corto casos nuevos](https://raw.githubusercontent.com/AP-2020-1S/covid-19-sandaye/master/Final%20Analitica/fig/5.medellin_corto%20_nuevoscasos.html?token=AQGJGC7VSEGSYS52FAPCLMS7KSCII)

### Definir fecha de pronostico 
```python
fecha = pd.Series.last_valid_index(real)+datetime.timedelta(days=1)
rango = pd.date_range(start=fecha, periods=15, freq='d')
```

### pronostico de mejor modelo a 7 días

```python
#Ajuste de modelo a datos
modelo_arima.fit(real['CASOS'])

#Pronostico
pronostico_casos = modelo_arima.predict(n_periods= 15, return_conf_int = True)
#prediccion_casos
pronostico = pd.DataFrame(pronostico_casos[0], index = rango, columns = ['Pronostico'])
```

### Intervalos de confianza de predicción

```python
banda_baja = pd.Series(pronostico_casos[1][:, 0], index = rango)
banda_alta = pd.Series(pronostico_casos[1][:, 1], index = rango)
```

### Pronostico
```python
fig_pcasos = go.Figure()
fig_pcasos.add_trace(go.Scatter(x=real.index, y=real['CASOS'], mode='lines', line={'color': 'salmon'}, name='Casos Diarios Reales'))
fig_pcasos.add_trace(go.Scatter(x= pronostico.index, y=pronostico['Pronostico'],mode='lines+markers',line={'color': 'rebeccapurple'},name='Pronostico Casos Diarios'))
fig_pcasos.add_trace(go.Scatter(x=rango, y=banda_baja,mode='lines', line={'color': ' powderblue'},name='band_conf_low'))
fig_pcasos.add_trace(go.Scatter(x=rango, y=banda_alta,mode='lines', line={'color': ' powderblue'},name='band_conf_up'))
fig_pcasos.update_layout(
        hovermode='x',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color='#3F3F3F',
        ),
        legend=dict(
            x=0.02,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color= '#474747',
            ),
            bgcolor='#FFFFFF',
            borderwidth=3
        ),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#F4EEEF',
        margin=dict(l=0, 
                    r=0, 
                    t=0, 
                    b=0
                    ),
      

    )
fig_pcasos.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#D9D8D8')
fig_pcasos.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#D9D8D8')

fig_pcasos.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#474747')
fig_pcasos.show()
fig.show()
```
![corto casos nuevos](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/fig/5.medellin_corto_pronostico_nuevoscasos.html)

### Construcción del modelo AUTO-ARIMA para los recuperados

...
### Construcción del modelo AUTO-ARIMA para los fallecidos día

...
### Construcción del modelo AUTO-ARIMA para los fallecidos acumulados
...
### Construcción del modelo AUTO-ARIMA para los activos
...
### Construcción del modelo AUTO-ARIMA para los confirmados
..

## BOGOTÁ
---
## CALI
---
## BARRANQUILLA
---
## CARTAGENA
---








## Referencias

Apellido, (Inicial nombre). ORGANIZACIÓN. Titulo articulo. año. Tomado de: URL

ejemplo:

González, M. EAFIT. Modelo SEIR para Colombia: Medidas de mitigación del virus. 2020 . Tomado de : https://www.eafit.edu.co/escuelas/economiayfinanzas/cief/Documents/informe-especial-2020-abril-2.pdf  




<!--encabezados-->


# Titulo 1
## Titulo 2
### Titulo 3
#### Titulo 4
##### Titulo 5
###### Titulo 6

<!--listas ordenadas-->
* uno
* dos
* tres

1. uno
    1. uno dos
2. dos
3. tres

[google.com](https://www.google.com "custom title")



---

```python
print("hello word")
```

```html
<h1>hello word</h1>
```

|titulo 1   | titulo 2  | titulo 3|  
|-----------|-----------|---------|
|texto      |  texto    | texto   |



![visual studio code logo](data:image/jpeg;base64,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 "visual code")


<!-- GitHub MarDonw-->

* [x] tarea 1
* [] tarea 2
* [] tarea 3
* [x] tarea 4