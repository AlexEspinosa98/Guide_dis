import sqlite3







""" # Eliminar la abse de datos 

# Conectarse a la base de datos
conexion = sqlite3.connect('test.db')  # Reemplaza con la ruta y nombre de tu base de datos

# Crear un cursor
cursor = conexion.cursor()

# Obtener los nombres de todas las tablas
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tablas = cursor.fetchall()

# Eliminar cada tabla
for tabla in tablas:
    nombre_tabla = tabla[0]
    cursor.execute(f"DROP TABLE {nombre_tabla};")

# Confirmar los cambios
conexion.commit()

# Cerrar la conexión
conexion.close()




 

#import sqlite3
#CREACIÓN  de las tablas en la base de datos
# Conectarse a la base de datos
conexion = sqlite3.connect('test.db')  # Reemplaza con la ruta y nombre de tu base de datos

# Crear un cursor
cursor = conexion.cursor()

# Crear una tabla
cursor.execute('''CREATE TABLE IF NOT EXISTS registro_carpeta (
                    id INTEGER PRIMARY KEY,
                    nombre_carpeta TEXT,
                    numero_de_imagenes INTEGER,
                    ruta_carpeta TEXT
                );''')

# Crear otra tabla
cursor.execute('''CREATE TABLE IF NOT EXISTS tabla_imagenes (
                    id INTEGER PRIMARY KEY,
                    nombre_imagen TEXT,
                    codigo_carpeta INTEGER,
                    cantidad_detect INTEGER
                );''')

cursor.execute('''CREATE TABLE IF NOT EXISTS resultado_imagen (
                    id INTEGER PRIMARY KEY,
                    pixel_min TEXT,
                    pixel_max TEXT,
                    latitud TEXT,
                    longitud TEXT,
                    nom_imagen TEXT,
                    codigo_carpeta INTEGER
                );''')


# Confirmar los cambios
conexion.commit()

# Verificar las tablas existentes
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tablas = cursor.fetchall()

# Imprimir los nombres de las tablas
for tabla in tablas:
    print(tabla[0])

# Cerrar la conexión
conexion.close() """

# Consulta en la base de datos columnas
""" conn = sqlite3.connect('test.db')
cursor = conn.cursor()

# Obtener el nombre de las tablas
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tablas = cursor.fetchall()

# Obtener las columnas de cada tabla
for tabla in tablas:
    cursor.execute(f"PRAGMA table_info({tabla[0]})")
    columnas = cursor.fetchall()

    # Imprimir el nombre de las columnas de la tabla
    print(f"Columnas de la tabla {tabla[0]}:")
    for columna in columnas:
        print(columna[1])
    print()

# Cerrar la conexión
conn.close()

 """

# consultar todas las columnas de la base de datos
conn = sqlite3.connect('test.db')
cursor = conn.cursor()

# Ejecutar la consulta SELECT *
cursor.execute("SELECT * FROM registro_carpeta")

# Obtener los resultados de la consulta
resultados = cursor.fetchall()

# Recorrer los resultados e imprimir los valores
for fila in resultados:
    for valor in fila:
        print(valor)
    print()

# Cerrar la conexión
conn.close()


# ingresar datos en sqlite
""" 
conn = sqlite3.connect('test.db')
cursor = conn.cursor()

# Insertar datos en la tabla
var=["DJI_0010.JPG" ,"DJI_0010.JPG","DJI_0010.JPG","DJI_0010.JPG","DJI_0010.JPG","DJI_0010.JPG","DJI_0010.JPG"]
var2=2
var3="ejemplo/base/datos/imagenes"
cursor.execute("INSERT INTO registro_carpeta (nombre_carpeta, numero_de_imagenes,ruta_carpeta) VALUES (?, ?, ?)",
               (var, var2, var3))

# Guardar los cambios y cerrar la conexión
conn.commit()
conn.close()


 """

""" 
conn = sqlite3.connect('test.db')
cursor = conn.cursor()

# ID de la fila a eliminar
id_fila = 1

# Eliminar la fila por ID
cursor.execute("DELETE FROM registro_carpeta WHERE id = ?", (2,))

# Guardar los cambios y cerrar la conexión
conn.commit()
conn.close()
 """


""" conn = sqlite3.connect('test.db')
cursor = conn.cursor()

# Ejecutar la consulta SELECT *
cursor.execute("SELECT * FROM registro_carpeta")

# Obtener los resultados de la consulta
resultados = cursor.fetchall()

# Recorrer los resultados e imprimir los valores
for fila in resultados:
    for valor in fila:
        print(valor)
    print()

# Cerrar la conexión
conn.close() """