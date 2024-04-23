# SVM-Traffic-Classifier
Machine Learning model for classifying data traffic by SVM.

# Creación de Entorno Virtual

### Nota: Es importante tener instalado Python 3.8 o una versión posterior para poder ejecutar este proyecto.

## Windows

Para crear un entorno virtual en Windows, sigue estos pasos:

1. **Instalar `virtualenv`**: Si aún no tienes instalado `virtualenv`, puedes instalarlo utilizando `pip`:

    ```bash
    pip install virtualenv
    ```

2. **Crear el Entorno Virtual**: Abre tu terminal y navega hasta el directorio donde deseas crear el entorno virtual. Luego, ejecuta el siguiente comando para crear un nuevo entorno virtual:

    ```bash
    python -m venv venv
    ```

    Esto creará una carpeta llamada `venv` en tu directorio actual que contendrá el entorno virtual.

3. **Activar el Entorno Virtual**: Para activar el entorno virtual, ejecuta el siguiente comando:

    ```bash
    venv\Scripts\activate
    ```

    Ahora estarás dentro del entorno virtual y podrás instalar paquetes específicos para tu proyecto sin afectar el entorno global de Python.

## Linux

Para crear un entorno virtual en Linux, sigue estos pasos:

1. **Instalar `virtualenv`**: Si aún no tienes instalado `virtualenv`, puedes instalarlo utilizando `pip`:

    ```bash
    pip install virtualenv
    ```

2. **Crear el Entorno Virtual**: Abre tu terminal y navega hasta el directorio donde deseas crear el entorno virtual. Luego, ejecuta el siguiente comando para crear un nuevo entorno virtual:

    ```bash
    python -m venv venv
    ```

    Esto creará una carpeta llamada `venv` en tu directorio actual que contendrá el entorno virtual.

3. **Activar el Entorno Virtual**: Para activar el entorno virtual, ejecuta el siguiente comando:

    ```bash
    source venv/bin/activate
    ```

    Ahora estarás dentro del entorno virtual y podrás instalar paquetes específicos para tu proyecto sin afectar el entorno global de Python.

## Requerimientos 
1. **Instalar Requerimientos del Proyecto**: Con el entorno virtual activado, puedes instalar los requerimientos del proyecto utilizando el archivo `requirements.txt` si lo tienes. Ejecuta el siguiente comando:

    ```bash
    pip install -r requirements.txt
    ```

    Esto instalará todas las dependencias necesarias para el proyecto.
