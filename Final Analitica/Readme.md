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
1. ¿ Que modelo de pronóstico es ideal para proyectar información confiable en el corto o mediano plazo de los casos confirmados del  SARS-CoV-2 con sus respectivos desenlaces(recuperados o fallecidos)  para  las 5 principales ciudades de Colombia?

2. ¿Qué metodología es la adecuada para permitir que el producto de datos sea autosuficiente , confiable y duradero?

En la literatura existen diferentes técnicas de pronóstico para las series de tiempo como por ejemplo: regresiones lineales , modelos autorregresivos y de medias móviles (ARIMA) , redes neuronales , modelo SIR entre otros. Estas técnicas poseen características especiales y unos con mejores ajustes que otros según el problema que sea planteado. Adicionalmente existen metodologías para la construcción de proyectos como la CRISP-DM , metodología que permite que el producto tenga unas fases de construcción definidas obteniendo así un mejor resultado.

## OBJETIVO
El objetivo de este proyecto es la predicción de corto y mediano plazo del total de casos confirmados, los nuevos casos, los casos activos, recuperados y muertes para las 5 principales ciudades de Colombia, utilizando técnicas estadísticas, de inteligencia artificial o modelos híbridos.

# DATOS
La fuente principal de información proviene de la página oficial del gobierno nacional [datos.gov.co](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Casos-positivos-de-COVID-19-en-Colombia/gt2j-8ykr/data "DATOS COVID")  la cual contiene información actualizada dia a dia de los casos confirmados del  SARS-CoV-2 en el territorio nacional.

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

### Descarga de datos

Se construyo una rutina que permite la descarga automatica de los datos directamente desde la fuente del Gobierno Nacional, la cual genera un archivo de datos crudos en scv que son almacenados y usados por las rutinas posteriores.

_Ver rutina de descarga de datos en el siguiente link:_

[Rutina_descarga_datos](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/1.down_data.ipynb "Jupyter notebook")


### Limpieza de datos

Tras un análisis de las variables del arrchivo principal de datos se realiza la selección de variables a usar, la limpieza de los datos erróneos o faltantes, correción formato y fuente de datos, Homogeneización de caracteres, Selección de datos relevantes entre otras.

_Ver rutina de limpieza de datos en el siguiente link:_

[Rutina_limpieza_datos](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/2.Clear_data.ipynb "Jupyter notebook")


### Archivos de trabajo por ciudades
Dado que el proyecto busca el comportamiento del COVID-19 para las 5 ciudades principales de Colombia, se construye un archivo csv que contiene los datos relevantes para la construcción de modelos y pronostico de cada ciudad.

Para este ejercicio se toman las 5 ciudades con mayor número de habitantes según el [DANE](https://www.dane.gov.co/ "PAG_DANE"), por lo cúal son aquellas con mayor número de personas susceptible de contagio de SARS-CoV-2.

| Ciudad | Población|
|--------|----------|
|Bogotá |7.412.566|
|Medellín| 2.427.129|
|Cali| 2.227.642|
|Barranquilla| 1.206.319|
|Cartagena| 973.045| 

_Ver código de creación de archivos de trabajo en el siguiente link:_

[Crear_archivos_trabajo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/3.dataframe_ciudades.ipynb "Jupyter notebook")



## Análisis Exploratorio de los datos

Para el análisis de los datos se tomaron las variables que tenian mayor grado de completitud y un menor grado de problemas de calidad del datos, estos problemas fueron los siguientes:

* Registros con "fecha de recuperado" y "fecha de muerte"
* Registros vacíos en columna de "estado" y "atención"
* Registros de fecha sin formato fecha


| Categoría | Variable Usada |
|-----------|-----------------|
| Casos diarios | FECHA REPORTE WEB|
| Fallecidos diarios| FECHA DE MUERTE|
| Recuperados | FECHA RECUPERADO |

La tabla presenta las variables selecionadas para cada categoria de análisis, de estas se construyen las subsecuentes variables que responden a valores acumulados u operaciones de las mismas.

### 
situación de cada ciudad (grafica)

total infectados en tabla

muestras procesadas, influye en el pronostico







# METODOLOGIAS PROPUESTAS
Para  este proyecto se proponen las siguientes metodologías de pronóstico 

## 1. _Modelo de pronóstico  de corto  plazo_
## 1.1 _Modelo ARIMA_
En 1970, Box y Jenkins desarrollaron un cuerpo metodológico destinado a identificar, estimar y diagnosticar modelos dinámicos de series temporales en los que la variable tiempo juega un papel fundamental. Una parte importante de esta metodología está pensada para liberar al investigador económetra de la tarea de especificación de los modelos dejando que los propios datos temporales de la variable a estudiar nos indiquen las características de la estructura probabilística subyacente.

El inconveniente es que, al renunciar a la inclusión de un conjunto más amplio de variables explicativas, no atendemos a las relaciones que sin duda existen entre casi todas las variables económicas perdiendo capacidad de análisis al tiempo que renunciamos, implícitamente, al estudio teórico previo del fenómeno y a su indudable utilidad. Sin embargo , los modelos ARIMA son de gran utilidad en muchos campos. En este proyecto elegimos usar este modelo gracias a su gran potencial y simpleza tanto en interpretación como en aplicabilidad

* _Proceso estocástico y estacionariedad_

Un proceso estocástico es una sucesión de variables aleatorias Y ordenadas,pudiendo tomar t cualquier valor entre -infinito y infinito. Por ejemplo, la siguiente sucesión de variables aleatorias puede ser considerada como proceso estocástico:

<img aling="center" src="fig\Arima_estacionaridad.png"
     alt="arima"
     style="float: left; margin-right: 1000px;" />                                                         
  
Decimos que un proceso estocástico es estacionario si las funciones de distribución conjuntas son invariantes con respecto a un desplazamiento en el tiempo (variación de t). Es decir, considerando que t, t+1, t+2, ...., t+k reflejan períodos sucesivos:  
  
    


<img aling="center" src="fig\Arima_estacionnaridad2.png"
     alt="arima"
     style="float: left; margin-right: 1000px;" />

* _Especificación del  modelo ARIMA_

En su forma más general el modelo ARIMA(p,d,q) ARIMA(P,D,Q,)S podría escribirse como:
<img aling="center" src="fig\Arima_estacionaridad3.png"
     alt="arima"
     style="float: left; margin-right: 1000px;" />



Entendiendo que puede haber más de un proceso generador de la serie (en la parte regular y en la estacional) y escribiendo una combinación de los modelos MA(q) y AR(p) que han precisado de una serie de diferenciaciones "d" en la parte regular o "D" en la parte estacional para que fueran estacionarios. 

Variables a predecir por el modelo ARIMA nombrado anteriormente:

|  Variable             |    Tipo de dato         |  
|-----------------------|-------------------------|
|Casos Nuevos (diario)               |  Serie de Tiempo               |
|Casos Confirmados (Acumulado)  |  Serie de Tiempo                 |
|Casos Activos (Acumulado)        |  Serie de Tiempo               |
|Recuperados (Acumulado)   |  Serie de Tiempo             |
|Muertes (Acumulado)|  Serie de Tiempo             |
|Muertes (Diario)               |  Serie de Tiempo                |


## _2. Modelo de pronóstico  de mediano  plazo_
## _2.1 Modelo SIR_
Los modelos SIR fueron desarrollados por Kermack y McKendrick en 1927 y han sido aplicados en diversos escenarios de epidemias. Estos modelos estiman el número teórico de personas susceptibles de enfermar (susceptibles), el número de enfermos (infectados) y el número de personas que ya no pueden transmitir la enfermedad (Recuperados o fallecidos), en una población a lo largo del tiempo. Los supuestos básicos de los modelos SIR son: a. La población es homogénea y de tamaño fijo; b. En un momento dado, cada individuo sólo puede pertenecer a uno de los siguientes conjuntos: infectados, susceptibles o resistentes; c. La interacción entre los individuos es aleatoria, y; d. No hay intervención externa que cambie la tasa de contacto de la población. En estos modelos se asume que la población por estado (N) es constante y que el número de individuos susceptibles S(t), infectados I(t) y fallecidos R(t) son variables dependientes del tiempo, de manera que:

<img aling="center" src="fig\SIR.png"
     alt="arima"
     style="float: left; margin-right: 1000px;" />

Dado que el tamaño de la población es fijo, se puede reducir el sistema de ecuaciones a otro con dos ecuaciones, definiendo r(t)=1-s(t)-i(t). Los modelos se pueden establecer con indicadores previamente elaborados a partir del comportamiento del microorganismo estudiado y de sucesos previamente establecidos (brotes anteriores), en los cuales es clave precisar la patogenicidad, la duración media de la enfermedad, las tasas de interacción, la probabilidad de contagio, la tasa de recuperación, su letalidad y mortalidad en poblaciones definidas, así como un R0 (número básico de reproducción) y Rt (número de reemplazamiento). 

 Variables a predecir por el modelo SIR nombrado anteriormente:

|  Variable             |    Tipo de dato         |  
|-----------------------|-------------------------|
|Población Susceptible (Acumulado)              |  Serie de Tiempo               |
|Casos Infectados (Acumulado)  |  Serie de Tiempo                 |
|Casos Activos (Acumulado)        |  Serie de Tiempo               |
|Recuperados y Muertes (Acumulado)    |  Serie de Tiempo             |


# DESARROLLO DE LOS MODELOS
## _Esquema de desarrollo del modelo ARIMA_
<img aling="center" src="fig\ESQUEMAARIMA.png"
     alt="arima"
     style="float: left; margin-right: 1000px;" />

## _Código Python(Etapas)_

* _Etapa 1_

```python
import requests
import os.path
import os
import pandas as pd
import numpy as np
import requests
import os.path
import os
import datetime
import dateutil
import subprocess
import sys
import tempfile

!pip install unidecode
import unidecode

# data oficial
url_dataset = 'https://www.datos.gov.co/api/views/gt2j-8ykr/rows.csv?accessType=DOWNLOAD'
# muestras procesadas diarias
url_muestras_diarias = 'https://e.infogram.com/api/live/flex/638d656c-c77b-4326-97d3-e50cb410c6ab/8188140c-8352-4994-85e3-2100a4dbd9db?'

# datos de entrada disponibles en el directorio "../Input/".
'./'
if os.path.split(os.path.abspath('.'))[-1] == 'src':
    '../Input_data'
# datos de salida disponibles en el directorio "../Output/".
'./'
if os.path.split(os.path.abspath('.'))[-1] == 'src':
    '../Output'

    # crear archivo data_covid.csv
with requests.get(url_dataset) as dataset_oficial:
    with open(os.path.join('../Input_data', 'data_covid.csv'), 'wb') as dataset_file:
        dataset_file.write(dataset_oficial.content)

        # crear archivo muestras diarias procesadas
with requests.get(url_muestras_diarias) as dataset_muestras:
    with open(os.path.join('../Input_data', 'data_muestras_procesadas.csv'), 'wb') as dataset_file:
        dataset_file.write(dataset_muestras.content)

# Lee el archivo descargado
dfcovid = pd.read_csv(os.path.join('../Input_data', 'data_covid.csv'))

# nombre de atributo en mayuscula y sin acento
dfcovid.columns = [unidecode.unidecode(value).upper() for value in dfcovid.columns]

# Eliminar ciudades diferentes a las 5 seleccionadas:
dfcovid=dfcovid.drop(dfcovid[(dfcovid['CODIGO DIVIPOLA'] != 5001)&(dfcovid['CODIGO DIVIPOLA'] != 11001)&(dfcovid['CODIGO DIVIPOLA'] != 8001)&(dfcovid['CODIGO DIVIPOLA'] != 76001)&(dfcovid['CODIGO DIVIPOLA'] != 13001)].index)

#Eliminar columnas basura
dfcovid = dfcovid.drop(['NOMBRE GRUPO ETNICO','PERTENENCIA ETNICA'], axis=1)

#Convertir las columnas a tipo string para un facil manejo
dfcovid = dfcovid.astype({"DEPARTAMENTO O DISTRITO ": str, "TIPO RECUPERACION": str,"PAIS DE PROCEDENCIA": str,"ESTADO": str, "ATENCION": str})

#Transformar los datos de texto en mayusculas (PENDIENTE)
dfcovid['CIUDAD DE UBICACION'] = [unidecode.unidecode(value).upper() for value in dfcovid['CIUDAD DE UBICACION']]
dfcovid['DEPARTAMENTO O DISTRITO '] = [unidecode.unidecode(value).upper() for value in dfcovid['DEPARTAMENTO O DISTRITO ']]
dfcovid['ATENCION'] = [unidecode.unidecode(value).upper() for value in dfcovid['ATENCION']]
dfcovid['SEXO'] = [unidecode.unidecode(value).upper() for value in dfcovid['SEXO']]
dfcovid['TIPO'] = [unidecode.unidecode(value).upper() for value in dfcovid['TIPO']]
dfcovid['ESTADO'] = [unidecode.unidecode(value).upper() for value in dfcovid['ESTADO']]
dfcovid['PAIS DE PROCEDENCIA'] = [unidecode.unidecode(value).upper() for value in dfcovid['PAIS DE PROCEDENCIA']]
dfcovid['TIPO RECUPERACION'] = [unidecode.unidecode(value).upper() for value in dfcovid['TIPO RECUPERACION']]

#Convertir columnas a tipo DATETIME
dfcovid['FECHA DE NOTIFICACION'] = pd.to_datetime(dfcovid['FECHA DE NOTIFICACION'])
dfcovid['FECHA DE MUERTE'] = pd.to_datetime(dfcovid['FECHA DE MUERTE'])
dfcovid['FECHA DIAGNOSTICO'] = pd.to_datetime(dfcovid['FECHA DIAGNOSTICO'])
dfcovid['FECHA RECUPERADO'] = pd.to_datetime(dfcovid['FECHA RECUPERADO'])
dfcovid['FECHA REPORTE WEB'] = pd.to_datetime(dfcovid['FECHA REPORTE WEB'])

#Crea la base final de trabajo
dfcovid.to_csv(os.path.join('../Output', 'data_final.csv'), index=False)

#llama la base final de trabajo 
dfcovid_ciudades = pd.read_csv(os.path.join('../Output', 'data_final.csv'))

# Creación de base de trabajo por ciudad (ejemplo Medellín)

dfmedellin=dfcovid_ciudades[dfcovid_ciudades['CODIGO DIVIPOLA']==5001]

#DATOS DIARIOS
dfmedellin_nuevoscasos = pd.to_datetime(dfmedellin['FECHA REPORTE WEB'])
dfmedellin_nuevoscasos_daily = dfmedellin_nuevoscasos.groupby(dfmedellin_nuevoscasos.dt.floor('d')).size().reset_index(name='CASOS')
dfmedellin_nuevoscasos_daily.columns = ['FECHA', 'CASOS']

dfmedellin_muertes= pd.to_datetime(dfmedellin['FECHA DE MUERTE'])
dfmedellin_muertes_daily=dfmedellin_muertes.groupby(dfmedellin_muertes.dt.floor('d')).size().reset_index(name='FALLECIDOS')
dfmedellin_muertes_daily.columns = ['FECHA', 'FALLECIDOS']

dfmedellin_recu = dfmedellin[dfmedellin['FECHA DE MUERTE'].isna()] 
#se extrae las fechas de muertes que tambien estan en recuperados
dfmedellin_recu = pd.to_datetime(dfmedellin_recu['FECHA RECUPERADO'])
dfmedellin_recu_daily=dfmedellin_recu.groupby(dfmedellin_recu.dt.floor('d')).size().reset_index(name='RECUPERADOS')
dfmedellin_recu_daily.columns = ['FECHA', 'RECUPERADOS']

#BASE DE TRABAJO
db_medellin = pd.merge(dfmedellin_nuevoscasos_daily[['FECHA','CASOS']],dfmedellin_recu_daily[['FECHA','RECUPERADOS']],on='FECHA',how='left')
db_medellin = pd.merge(db_medellin[['FECHA','CASOS','RECUPERADOS']],dfmedellin_muertes_daily[['FECHA','FALLECIDOS']],on='FECHA',how='left')
db_medellin = db_medellin.fillna(0)
db_medellin['CASOS_ACUMM'] = db_medellin['CASOS'].cumsum()
db_medellin['RECUPERADOS_ACUMM'] = db_medellin['RECUPERADOS'].cumsum()
db_medellin['FALLECIDOS_ACUMM'] = db_medellin['FALLECIDOS'].cumsum()
db_medellin['ACTIVOS'] =db_medellin['CASOS_ACUMM']-db_medellin['RECUPERADOS_ACUMM']-db_medellin['FALLECIDOS_ACUMM']
db_medellin['ACTIVOS_T1']=db_medellin['ACTIVOS'].shift(1)
#db_medellin
db_medellin.to_csv(os.path.join('../Output', 'data_medellin.csv'), index=False)
```
* _Etapa 2 y 3_
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

# Datos Nuevos casos Medellín
df = pd.read_csv(os.path.join('../Output', 'data_medellin.csv'))
df = df[['FECHA','CASOS']]
df2 = pd.to_datetime(df['FECHA'])
df.index = df2
data_nuevoscasos = df.drop(['FECHA'], axis=1)

# Datos Recuperados Acumulados Medellín
df1 = pd.read_csv(os.path.join('../Output', 'data_medellin.csv'))
df1 = df1[['FECHA','RECUPERADOS_ACUMM']]
df3 = pd.to_datetime(df1['FECHA'])
df1.index = df3
data_recuperados = df1.drop(['FECHA'], axis=1)

# Datos Muertes diarias Medellín
df4 = pd.read_csv(os.path.join('../Output', 'data_medellin.csv'))
df4 = df4[['FECHA','FALLECIDOS']]
df5 = pd.to_datetime(df4['FECHA'])
df4.index = df5
data_fallecidos_diarios = df4.drop(['FECHA'], axis=1)

# Datos Muertes Acumuladas Medellín
df4 = pd.read_csv(os.path.join('../Output', 'data_medellin.csv'))
df4 = df4[['FECHA','FALLECIDOS_ACUMM']]
df5 = pd.to_datetime(df4['FECHA'])
df4.index = df5
data_fallecidos = df4.drop(['FECHA'], axis=1)

# Datos Activos Medellín
df5 = pd.read_csv(os.path.join('../Output', 'data_medellin.csv'))
df5 = df5[['FECHA','ACTIVOS']]
df6 = pd.to_datetime(df5['FECHA'])
df5.index = df6
data_activos = df5.drop(['FECHA'], axis=1)

# Datos Confirmados Medellín
df7 = pd.read_csv(os.path.join('../Output', 'data_medellin.csv'))
df7 = df7[['FECHA','CASOS_ACUMM']]
df8 = pd.to_datetime(df7['FECHA'])
df7.index = df8
data_confirmados = df7.drop(['FECHA'], axis=1)

# Construcción modelo AUTO-ARIMA para Nuevos Casos Diarios

# Data, data_train y data_test
real = data_nuevoscasos
train = data_nuevoscasos[:int(len(data_nuevoscasos)*(1-7/len(data_nuevoscasos)))]
test = data_nuevoscasos[-7:]

# definición del medelo -proceso de ciclo de simulación
modelo_arima = auto_arima(train, start_p=1, d=None, start_q=1, 
                          max_p=12, max_d=8, max_q=12,
                          start_P=1, D=None, start_Q=1, 
                          max_P=12, max_D=8, max_Q=12, max_order=None, m=7, seasonal=True,
                          trace=True, supress_warnings=True, stepwise=True, random_state=20, n_fits=50)


# Resultados del mejor modelo
modelo_arima.summary()


#Diagnostico Modelo
modelo_arima.plot_diagnostics(figsize=(14, 8))
plt.show()
```
* _Etapa 4_
```python
#test predicción con Auto Arima
test_prediccion_nuevoscasos = pd.DataFrame(modelo_arima.predict(n_periods = len(test)), index = test.index)
test_prediccion_nuevoscasos.columns = ['Test predicción nuevos casos']
#test_prediccion_nuevoscasos

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

fig.write_html('../Output/med_test_corto_nuev_casos.html')
```
## _Esquema de desarrollo del modelo SIR_
<img aling="center" src="fig\ESQUEMASIR.png"
     alt="arima"
     style="float: left; margin-right: 1000px;" />


## _Código Python(Etapas)_

* _Etapa 1_

Ver la etapa 1 del esquema de desarrollo ARIMA , funciona exactamente de la misma manera.

* _Etapa 2 y 3_

```python
import numpy as np
import pandas as pd
import os
import os.path
import datetime
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import plotly.offline as py
!pip install cufflinks
import cufflinks as cf
!pip install plotly==4.9.0
import plotly.graph_objects as go
from sklearn import metrics
from sklearn.metrics import mean_squared_error

#DataFrame de Activos ciudad de Medellín
df = pd.read_csv(os.path.join('../Output', 'data_medellin.csv'))
df = df[['FECHA','ACTIVOS']]
df2 = pd.to_datetime(df['FECHA'])
df.index = df2
data_activos = df.drop(['FECHA'], axis=1)

#MODELO SIR PARA VARIOS BETA Y GAMMA

def modelo_SIR(beta,gamma,t):
    N=100000
    I0, R0=1, 0
    S0=N - I0 - R0
    
    # Las ecuaciones diferenciales del modelo SIR
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
# Vector de las condiciones iniciales
    y0 = S0, I0, R0
# Resolver el sistema de ecuaciones diferenciales, en la secuencia de días que ya definimos
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return (I)

beta_x = 0.01
gamma_x = 0.015
mse_x = 1000000000000

real=data_activos['ACTIVOS']
t = np.linspace(0, len(data_activos), len(data_activos))


for beta in np.arange(0.01,0.3,0.01):
    for gamma in np.arange(0.015,0.3,0.005):
        I = modelo_SIR(beta,gamma,t)
        mse = mean_squared_error(real,I)
        if mse < mse_x:
            beta_x = beta
            gamma_x = gamma
            mse_x = mse
print(beta_x,gamma_x,mse_x)

```

* _Etapa 4_

```python

#Predicción con el mejor modelo
N =   100000
I0, R0 = 1, 0
S0 = N - I0 - R0
beta, gamma = beta_x, gamma_x

t1 = np.linspace(0, len(data_activos)+90, len(data_activos)+90)

def deriv(y, t1, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

y0 = S0, I0, R0

ret = odeint(deriv, y0, t1, args=(N, beta, gamma))
S, I, R = ret.T

fecha_serie = pd.Series.first_valid_index(data_activos)
rango_serie = pd.date_range(start=fecha_serie, periods=len(data_activos)+90, freq='d')

# grafica de la predicción
fig = go.Figure()
fig.add_trace(go.Scatter(x=rango_serie, y=S,mode='lines',name='Susceptible'))
fig.add_trace(go.Scatter(x=rango_serie, y=R,mode='lines',name='Recuperada y fallecidos'))
fig.add_trace(go.Scatter(x=rango_serie, y=I,mode='lines',name='Infectada'))
fig.add_trace(go.Scatter(x=rango_serie, y=data_activos['ACTIVOS'],mode='lines',name='Activos Reales'))
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

fig.write_html('../Templates/med_forecast_SIR.html')
```

## Referencias











 




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