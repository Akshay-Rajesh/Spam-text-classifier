# Slim version for faster
FROM python:3.10-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia el archivo requirements.txt en el directorio de trabajo actual
COPY requirements.txt .

# Instala las dependencias de Python especificadas en el archivo requirements.txt
# Se usa --no-cache-dir para no almacenar el caché de las descargas, reduciendo el tamaño de la imagen
RUN pip install --no-cache-dir -r requirements.txt

# Una vez instaladas las dependencias, copia el resto del contenido del directorio del proyecto al contenedor
COPY . .

# Expone el puerto 80 para permitir la comunicación al contenedor
EXPOSE 80

# Define una variable de entorno
ENV NAME World

# El comando predeterminado para ejecutar cuando se inicia el contenedor
CMD ["python", "predict.py"]

