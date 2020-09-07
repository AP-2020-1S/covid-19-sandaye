
# PRONÓSTICO DE LA EVOLUCIÓN DE CASOS CONFIRMADOS, RECUPERADOS Y MUERTES POR SARS-COV-2 PARA LAS 5 PRINCIPALES CIUDADES DE COLOMBIA

##### [Dashboard](https://meet.google.com/linkredirect?authuser=0&dest=https%3A%2F%2Fsangonzalez.github.io%2F "Abrir Dashboard")

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


_Ver código exploratio de datos en el siguiente link:_

[Exploratorio de datos](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/4.Exploratorio.ipynb "Jupyter notebook")



### Comportamiento SARS-COV-2 por cada ciudad principal

Se evalua las categorias de análisis en cada una de las ciudades principales, buscando comportamientos que permitan entender de forma general el avance del virus por cada ciudad. Los patrones que se identifican en caunto al comportamiento de los "picos" de la pandemia son claves para delimitar los datos de prueba y entrenamiento para la construcción de un modelo que permita un pronostico confiable.

### Tabla de resumen Covid para las cinco ciudades principales
* _Con corte al 5 de septiembre del 2020_

![Tabla resumen de ciudades - Covid](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/fig/tabla_total_ciudades.png)


#### Medellín

![Comportamiento SARS-COV-2](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/fig/totales_medellin.png)


#### Bogotá

![Comportamiento SARS-COV-2](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/fig/totales_bogota.png)


#### Cali

![Comportamiento SARS-COV-2](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/fig/totales_cali.png)


#### Barranquilla

![Comportamiento SARS-COV-2](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/fig/totales_barranquilla.png)



#### Cartagena

![Comportamiento SARS-COV-2](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/fig/totales_cartagena.png)


### Muestras procesadas

![Muestras diarias procesadas](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/fig/puebas_procesadas_diarias.png)

En los últimos 20 días, con corte al 5 de septiembre, se ha observado una disminución conciderable en el número de pruebas diarias realizadas, esto es, pasaron de 40.000 prubeas diarias a cerca de 25.000 diarias, lo que implica una reducción cercana al 40%.

Este es un dato relevante en el análsis de los resultados de los modelos de predicción de corto y mediano plazo, dado que dicha reducción también se ve reflejada en la tendencia descendente de los casos confirmados y fallecidos. Por lo que es muy problable que exista un subregistro en la base de datos y no reflejen de forma exacta la realidad de la pandemia.  


# METODOLOGIAS PROPUESTAS
Para  este proyecto se proponen las siguientes metodologías de pronóstico 

## _Modelo de pronóstico  de corto  plazo_
###  _Modelo ARIMA_

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


## _Modelo de pronóstico  de mediano  plazo_
### _Modelo SIR_
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

  

![Esquema](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/fig/ESQUEMAARIMA.PNG)

## _Código Python(Etapas)_

Toda la rutina se ejecuta de forma automatica y en orden desde el archivo [RUN.ipynb](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/RUN.ipynb "Jupyter notebook"), que contiene todas las rutinas del proyecto.

* _Etapa 1_: Actualización de las series de tiempo historicas con una única rutina

    * Primero se hace la descarga de datos: 
    
        _Ver rutina de descarga de datos en el siguiente link:_
    
        [Rutina_descarga_datos](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/1.down_data.ipynb "Jupyter notebook")

    * Posteriormente se ejecuta la rutina de limpieza de datos:
        _Ver rutina de limpieza de datos en el siguiente link:_

        [Rutina_limpieza_datos](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/2.Clear_data.ipynb "Jupyter notebook")

    * Por último se crean los archivos de trabajo:
        _Ver código de creación de archivos de trabajo en el siguiente link:_

        [Crear_archivos_trabajo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/3.dataframe_ciudades.ipynb "Jupyter notebook")

* _Etapa 2 y 3_: Auto-ajuste del modelo ARIMA a la serie de tiempo

    * A continuación se presentan las rutinas de auto-ajuste de los modelos tipo ARIMA para los datos historicos del SARS-COV-2 de las cinco principales ciudades:
    * Auto-ajuste modelo ARIMA para Medellín

        [Modelo Medellín corto plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/5.medellin_corto.ipynb "Jupyter notebook")

    * Auto-ajuste modelo ARIMA para Bogotá
        
        [Modelo Bogotá corto plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/6.bogota_corto.ipynb "Jupyter notebook")

    * Auto-ajuste modelo ARIMA para Cali
        
        [Modelo Cali corto plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/7.cali_corto.ipynb "Jupyter notebook")

    * Auto-ajuste modelo ARIMA para Barranquilla
        
        [Modelo Barranquilla corto plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/8.barranquilla_corto.ipynb "Jupyter notebook")

    * Auto-ajuste modelo ARIMA para Cartagena
        
        [Modelo Cartagena corto plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/9.cartagena_corto.ipynb "Jupyter notebook")



* _Etapa 4_: Pronóstico 

    * A continuación se presentan las rutinas de pronóstico del modelo ARIMA para los casos nuevos, confirmados, recuperados, fallecidos y activos de SARS-COV-2 para las cinco principales ciudades, usando el mejor modelo encontrado en la etapa anterior.
    
    * Pronóstico modelo ARIMA para Medellín

        [Modelo Medellín corto plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/5.medellin_corto.ipynb "Jupyter notebook")

    * Pronóstico modelo ARIMA para Bogotá
        
        [Modelo Bogotá corto plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/6.bogota_corto.ipynb "Jupyter notebook")

    * Pronóstico modelo ARIMA para Cali
        
        [Modelo Cali corto plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/7.cali_corto.ipynb "Jupyter notebook")

    * Pronóstico modelo ARIMA para Barranquilla
        
        [Modelo Barranquilla corto plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/8.barranquilla_corto.ipynb "Jupyter notebook")

    * Pronóstico modelo ARIMA para Cartagena
        
        [Modelo Cartagena corto plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/9.cartagena_corto.ipynb "Jupyter notebook")


## _Esquema de desarrollo del modelo SIR_

![Esquema_sir](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/fig/ESQUEMASIR.PNG)

## _Código Python(Etapas)_

Toda la rutina se ejecuta de forma automatica y en orden desde el archivo [RUN.ipynb](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/RUN.ipynb "Jupyter notebook"), que contiene todas las rutinas del proyecto.

* _Etapa 1_: Actualización de las series de tiempo historicas con una única rutina

    * Primero se hace la descarga de datos: 
    
        _Ver rutina de descarga de datos en el siguiente link:_
    
    [Rutina_descarga_datos](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/1.down_data.ipynb "Jupyter notebook")

    * Posteriormente se ejecuta la rutina de limpieza de datos:
        _Ver rutina de limpieza de datos en el siguiente link:_

    [Rutina_limpieza_datos](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/2.Clear_data.ipynb "Jupyter notebook")

    * Por último se crean los archivos de trabajo:
        _Ver código de creación de archivos de trabajo en el siguiente link:_

    [Crear_archivos_trabajo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/3.dataframe_ciudades.ipynb "Jupyter notebook")


* _Etapa 2 y 3_: Ajuste del modelo SIR a la serie de tiempo

    * A continuación se presentan las rutinas de ajuste deL modelo SIR para los datos historicos del SARS-COV-2 de las cinco principales ciudades, donde se construyo una rutina para la optimización de los parámetros "beta" y "gamma", esto se logro calculando el error cuadratico medio "mse" de la serie origanl y del modelo SIR, es decir que, se hizo una comparación entre varios modelos SIR variando estos párametros en una rango extenso por el método de tanteo para encontrar el menor error entre el modelo SIR y los datos originales(reales).

    * Auto-ajuste modelo SIR para Medellín

        [Modelo Medellín mediano plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/10.medellin_SIR_mediano.ipynb "Jupyter notebook")

    * Auto-ajuste modelo SIR para Bogotá
        
        [Modelo Bogotá mediano plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/11.bogota_SIR_mediano.ipynb "Jupyter notebook")

    * Auto-ajuste modelo SIR para Cali
        
        [Modelo Cali mediano plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/12.cali_SIR_mediano.ipynb "Jupyter notebook")

    * Auto-ajuste modelo SIR para Barranquilla
        
        [Modelo Barranquilla mediano plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/13.barranquilla_SIR_mediano.ipynb "Jupyter notebook")

    * Auto-ajuste modelo SIR para Cartagena
        
        [Modelo Cartagena mediano plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/14.cartagena_SIR_mediano.ipynb "Jupyter notebook")

* _Etapa 4_: Pronóstico 

    * A continuación se presentan las rutinas de pronóstico del modelo SIR para los casos nuevos, confirmados, recuperados, fallecidos y activos de SARS-COV-2 para las cinco principales ciudades, usando los mejores beta y gamma encontrados en la etapa anterior.
    
   * Auto-ajuste modelo SIR para Medellín

        [Modelo Medellín mediano plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/10.medellin_SIR_mediano.ipynb "Jupyter notebook")

    * Auto-ajuste modelo SIR para Bogotá
        
        [Modelo Bogotá mediano plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/11.bogota_SIR_mediano.ipynb "Jupyter notebook")

    * Auto-ajuste modelo SIR para Cali
        
        [Modelo Cali mediano plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/12.cali_SIR_mediano.ipynb "Jupyter notebook")

    * Auto-ajuste modelo SIR para Barranquilla
        
        [Modelo Barranquilla mediano plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/13.barranquilla_SIR_mediano.ipynb "Jupyter notebook")

    * Auto-ajuste modelo SIR para Cartagena
        
        [Modelo Cartagena mediano plazo](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/14.cartagena_SIR_mediano.ipynb "Jupyter notebook")-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/9.cartagena_corto.ipynb "Jupyter notebook")



# DESPLIEGUE


## Manejo del repositorio
* Se debe clonar el repositorio localmente.
En el siguiente link se encuentra el repositorio que contiene el proyecto de predicción para el SARS-COV-2: [Repositorio](https://meet.google.com/linkredirect?authuser=0&dest=https%3A%2F%2Fgithub.com%2FAP-2020-1S%2Fcovid-19-sandaye "Abrir")

* Abrir la carpeta "Final análitica" del repositorio y en esta encontrada el archivo "Readme.md", en el cual esta el desarrollo de los modelos de predicción para el SARS-COV-2.
* Abrir la carpeta "Notebooks" contenido en la carpeta "Final análitica".
### Función de las carpetas dentro del repositorio
* Final Análitica: contienen las carpetas "Input_data", "Notebooks", "Output", "Templates", "Fig y el archivo "Readme.md".
* Input_data: Se almacenan los archivos tipo csv con los datos crudos descargados de la fuente.
* Notebooks: Continene todas las rutinas de código en Python para el desarrollo de los modelos de predicción.
* Output: Se almacenan los archivos que han sido transformados y procesados con las rutinas de código.
* Templates: Contiene los archivos insumo del dashboard.
* Fig: Contienen las imagenes que se usan para la elaboración del archivo "Readme.md", donde esta contenido el análisis completo de los modelos.

En esta carpeta encontrada todas las rutinas del desarrollo de los modelos de predicción, estas pueden ser ejecutadas como se indica en el siguiente paso.

## Ejecutar las rutinas
Toda la rutina se ejecuta de forma automatica y en orden desde el archivo [RUN.ipynb](https://github.com/AP-2020-1S/covid-19-sandaye/blob/master/Final%20Analitica/Notebooks/RUN.ipynb "Jupyter notebook"), que contiene todas las rutinas del proyecto.

Despues de que el archivo Run.ipynb ejecute todas las rutinas se actualiza la información contenida en las carpetas que son insumo para el dashboard de forma automática.

## [Dashboard](https://meet.google.com/linkredirect?authuser=0&dest=https%3A%2F%2Fsangonzalez.github.io%2F "Abrir Dashboard")

En el despliegue del modelo se entregan los resultados obtenidos de la proyección de SARS-COV-2, para lo que se creó un dashboard utilizando HTML donde se visualiza la información y la proyección automáticamente de los nuevos casos, casos confirmados, recuperados, fallecidos y activos, para su lanzamiento se usó la plataforma GitHub Pages que permite albergar sitios web directamente desde un repositorio de GitHub.

### Contenido del Dashboard
* Banner para visuaizar por cada una de las 5 ciudades principales objeto de estudio.
* Datos relevantes de cada ciudad respecto al SARS-COV-2: Total acumulado casos confirmados, total acumulado casos recuperados y total acumulado casos fallecidos.
* Graficos interactivos con los pronósticos a corto y medianp plazo para el comportamiento del SARS-COV-2.
    * Pronóstico de mediano plazo usando el modelo SIR (90 días)
    * Pronóstico de corto plazo usando el modelo ARIMA (15 días)

La visualización de dashboard se puede realizar en el siguiente link: [Dashboard](https://meet.google.com/linkredirect?authuser=0&dest=https%3A%2F%2Fsangonzalez.github.io%2F "Abrir Dashboard")

La rutina completa es ejecutable todos los días despues de las 10 pm, hora en la que se actualizan los datos historicos de las 5 cuidades pronósticadas.

Los modelos de corto y mediano plazo estan sujetos a recalibración debido a cambios en las dinámicas del pais, como la de implementación de politicas de apertura, cierre, avance del virus y comportamiento social.

la vida útil de los modelos de predicción estan determinados por factores exogénos , tales como:
* Encontrar la vacuna que termiene con el SARS-COV-2.
* Cuando se adquiera inmunidad de la población por su contagio total.
* Por politicas de país que ya no permitan la actualización de los datos.


# Referencias

Instituto Nacional de Salud.(2020). Modelo de Transmisión de Coronavirus COVID-19, Escenarios para Colombia. Tomado de : https://www.ins.gov.co/Direcciones/ONS/SiteAssets/Modelo%20COVID-19%20Colombia%20INS_v5.pdf

Gobierno Nacional de Colombia.Datos abiertos: https://www.datos.gov.co/Salud-y-Protecci-n-Social/Casos-positivos-de-COVID-19-en-Colombia/gt2j-8ykr/data

Manrique, F., et al.(2020). Modelo SIR de la pandemia de Covid-19 en Colombia. Revista de Salud Pública, 2020, vol. 22, p. 1-9. http://dx.doi.org/10.15446/rsap.v22.85977

De Arce, R., Mahía,R.(2003). Modelos Arima [Archivo PDF]. https://d1wqtxts1xzle7.cloudfront.net/53321017/Box-Jenkins.PDF?1496089235=&response-content-disposition=inline%3B+filename%3DMODELOS_ARIMA.pdf&Expires=1599448076&Signature=KXM6GabDJdqsv01R5KF2s3tpON82zSJne9rwwloQQht36d75UWEjhJ5WvMmo-fGaS7tZPpvXgi2TeOs29qwDAMJ1o3~K7AsnrsLS1H3AR-TgFyHslILSqrlfIyFS9LU-6DXmZV6YGF7gxJMp~RX63PxcS1QA-MeVKkWZxz9EYLBHTel44BhN9OjOum575M0CpCurotPNwIAgw55x2HQdgyFqVYHdFF5XVf58ONcyvSxNQJD0bgOCpPPgjaIzEd7tOaVPFU~hP5GfwLTEOSakQM9GC31Q0S6RDovRIXxG8ppnarroKKGgP2tIavth9iaiPd~xZaIRpTjD4az1ESR1UA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA

jdvelasq.github.io/courses/(2019). Analítica Predictiva https://jdvelasq.github.io/courses/

