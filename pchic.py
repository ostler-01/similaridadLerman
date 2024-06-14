import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import math
from numpy.ma import count

pd.set_option('display.float_format', '{:.7f}'.format)

# Variables globales
contenedor_de_la_matriz = []
contenedor_matrix_aa = []
contenedor_lista_matriz = []
contenedor_nodos = []
contenedor_x = []
xx = []
yy = []
contenedor_dxf = []
contenedor_df_concatenado = []
contenedor_valor = []
resultados = []
contendor_diccionario = []
contenedor_f = []
contenedor_sumaI = []
contenedor_rk = []
contenedor__sk = []
contenedor_skk = []
contenedor_rkk = []
fi = []
Ii = []
con_suma = []
contenedor_v = []
contenedor_niveles = []
contenedor_datos_niveles = []
nuevo_vector_final=[]
segundo_maximo= []
maximo=[]
p= int

def calcular_similitud(base):
    global dfn
    global column_labels
    
    contenedor_indice_similaridad = []
    contenedor_matriz = []
    card_values = []
    
    matriz = np.zeros((len(base.columns), len(base.columns)))

    for i in range(len(base.columns)):
        for j in range(i + 1, len(base.columns)):
            A = base.columns[i]
            B = base.columns[j]
            ai = base[A]
            aj = base[B]
            card = np.sum((ai * aj))
            card_values.append(round(card))
            n = len(base)
            n_ai = np.sum((ai))
            n_aj = np.sum((aj))
            kc = (card - (n_ai * n_aj) / n) / np.sqrt((n_ai * n_aj) / n)
            sim = norm.cdf(kc)
            contenedor_indice_similaridad.append((A, B, card, kc, sim))
            contenedor_lista_matriz.append((A, B, round(sim, 2)))

            matriz[i, j] = sim
            matriz[j, i] = sim

    contenedor_matriz.append(matriz)
    column_labels = base.columns.tolist()
    dfn = pd.DataFrame(matriz)
    
    encabezados = column_labels
    dfn.columns = encabezados
    dfn.index = encabezados
    segundo_maximo = np.amax(matriz)
    contenedor_de_la_matriz.append(dfn)
    

    data_sim = pd.DataFrame(contenedor_indice_similaridad, columns=['var1', 'var2', 'card(ai ∩ aj)', 'kc', 's(ai,aj)'])

    resultado_text.delete(1.0, tk.END)
    resultado_text.insert(tk.END, "VALORES DE COPRESENCIAS ESTANDARIZADAS E ÍNDICES DE SIMILARIDAD\n")
    resultado_text.insert(tk.END, str(data_sim) + "\n")
    resultado_text.insert(tk.END, "Matriz de Similaridad:\n")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # Esta opción muestra la matriz completa
        resultado_text.insert(tk.END, str(dfn) + "\n")
    resultado_text.update_idletasks()
    
    # Contenedores
    valor_maximo = []
    contenedor_etiquetas1 = []

    segundo_maximo = np.amax(matriz)
    valor_maximo.append(segundo_maximo)

    data_sim = pd.DataFrame(contenedor_indice_similaridad, columns=['var1', 'var2', 'card(ai ∩ aj)', 'kc', 's(ai,aj)'])

    # Nivel Cero
    
    resultado_text.insert(tk.END, f"VARIABLES {column_labels}, VALOR DEL NIVEL: {segundo_maximo}\n")
    resultado_text.insert(tk.END, "--------------------------------------------------------------------------\n")
    

    # INICIO DE MATRIZ CUADRADA
    lista = np.array(contenedor_matriz).flatten().tolist()
    dimension = int(len(lista) ** 0.5)
    contenedor_matrix = []

    if dimension * dimension != len(lista):
        resultado_text.insert(tk.END, "La lista no tiene una dimensión cuadrada perfecta.\n")
    else:
        matriz_cuadrada = [[lista[i * dimension + j] for j in range(dimension)] for i in range(dimension)]

    for fila in matriz_cuadrada:
        contenedor_matrix.append(fila)

    df = pd.DataFrame(contenedor_matrix)
    aa = np.matrix(df)

    w = aa.shape[0]
    q = aa.shape[1]

    caracteres_eliminados = []

    for i in range(w):
        for j in range(q):
            indice_max = np.unravel_index(np.argmax(aa), aa.shape)
            fila_eliminada = np.ravel(aa[0:, indice_max[0]])
            nuevo_encabezado = np.delete(column_labels, indice_max, axis=0)
            vector1 = np.array(column_labels)
            vector2 = np.array(nuevo_encabezado)

            nuevo_vector = np.setdiff1d(vector1, vector2)
            resultado_vector = ','.join(nuevo_vector)
            vector3 = f"V{i}"
            nuevo_vector_final = np.insert(nuevo_encabezado, 0, vector3)

    resultado_text.insert(tk.END, f"VARIABLES {nuevo_vector_final},  VALOR DEL NIVEL: {segundo_maximo}\n")
    resultado_text.insert(tk.END, "--------------------------------------------------------------------------------j\n")
    
    # Agregar estas líneas aquí:
    variables_reales_v4 = [column_labels[i] for i in range(len(column_labels)) if column_labels[i] in nuevo_vector]
    resultado_text.insert(tk.END, f"Las variables reales contenidas en {vector3} son: {', '.join(variables_reales_v4)}\n")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # Esta opción muestra la matriz completa
        resultado_text.insert(tk.END, str(vector3) + "\n\n")
    resultado_text.update_idletasks()
    
    
   
    
    contenedor_datos_niveles.append((0, segundo_maximo, nuevo_vector_final, vector3, None))

 
  
    colum_eliminada = aa[:, indice_max[1]]
    columna_eliminada = np.transpose(colum_eliminada)
    fila_eliminada = aa[indice_max[0], :]

    indice_maxa = np.argmax(columna_eliminada)
    indice_mina = np.argmin(columna_eliminada)
    indice_maxb = np.argmax(fila_eliminada)
    indice_minb = np.argmin(fila_eliminada)

    for k in range(len(colum_eliminada)):
        vector1 = np.delete(colum_eliminada, [indice_maxa, indice_mina])
        vector2 = np.delete(fila_eliminada, [indice_maxb, indice_minb])
        vector3 = np.delete(colum_eliminada, [indice_maxa, indice_mina])
        maximos_por_posicion = np.maximum.reduce([vector1, vector2, vector3])

    p = (2 * 1)
    nuevo_nn = np.array(maximos_por_posicion ** (p))
    matrix_uni = np.delete(np.delete(aa, indice_max, axis=0), indice_max, axis=1)
    matriz_n = np.zeros((len(matrix_uni) + 1, len(matrix_uni) + 1))
    matriz_n[1:, 1:] = matrix_uni
    matriz_n[0, 1:] = nuevo_nn
    matriz_n[1:, 0] = nuevo_nn
    matriz_uni = matriz_n

    nueva_matriz = matriz_uni
    matriz_simetrica = np.array(nueva_matriz)
    maximo = np.amax(matriz_simetrica)

    nombres = [f'{nuevo_vector_final[i]}' for i in range(matriz_simetrica.shape[0])]
    contenedor_etiquetas1.append(nombres)

    dfss = pd.DataFrame(matriz_simetrica, index=nombres, columns=nombres)

    w1 = matriz_simetrica.shape[0]
    q1 = matriz_simetrica.shape[1]

    for i in range(w1):
        for j in range(q1):
            indice_max = np.unravel_index(np.argmax(matriz_simetrica), matriz_simetrica.shape)
            fila_eliminada = np.ravel(matriz_simetrica[0:, indice_max[0]])
            nuevo_encabezado = np.delete(nombres, indice_max, axis=0)
            vector1 = np.array(nombres)
            vector2 = np.array(nuevo_encabezado)

            nuevo_vector = np.setdiff1d(vector1, vector2)
            resultado_vector = ','.join(nuevo_vector)
            vector3 = f"A{i}"
            nuevo_vector_final = np.insert(nuevo_encabezado, 0, vector3)
   
            
    resultado_text.insert(tk.END, "MATRIZ NIVEL 1\n")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # Esta opción muestra la matriz completa
        resultado_text.insert(tk.END, str(dfss) + "\n")
    resultado_text.insert(tk.END, "----------------------------------------------------------\n")
    resultado_text.insert(tk.END, f"VARIABLES {nuevo_vector_final} = {vector3}\n")
    resultado_text.insert(tk.END, f"VARIABLES {nuevo_vector_final}, VALOR DEL NIVEL : {maximo}\n")
    variables_reales_v4 = [column_labels[i] for i in range(len(column_labels)) if column_labels[i] in nuevo_vector]
    resultado_text.insert(tk.END, f"Las variables reales contenidas en {vector3} son: {', '.join(variables_reales_v4)}\n")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # Esta opción muestra la matriz completa
        resultado_text.insert(tk.END, str(vector3) + "\n")
    resultado_text.update_idletasks()

    etiquetas = nombres
    etiquetasq = [etiqueta for etiqueta in etiquetas if etiqueta not in (nuevo_vector)]
    etiquetasq.insert(0, vector3)

    contenedor_var = []
    longitud = len(matriz_simetrica)
    contenedor0 = []

    for i in range(longitud):
        l =0
        k= 1
        # si las variables  (longitud) son mayores a  hacer matriz uni mayor a niveles
        while (len(matriz_uni) > 1):
            l =  l+1
            k = k+1
            
            # Verificar si la matriz es toda cero
            if np.all(matriz_uni == 0):
                break

            indice_max = np.unravel_index(np.argmax(matriz_uni), matriz_uni.shape)
            colum_eliminada = matriz_uni[:, indice_max[1]]
            columna_eliminada = np.transpose(colum_eliminada)
            fila_eliminada = matriz_uni[indice_max[0], :]

            indice_maxa = np.argmax(columna_eliminada)
            indice_mina = np.argmin(columna_eliminada)
            indice_maxb = np.argmax(fila_eliminada)
            indice_minb = np.argmin(fila_eliminada)

            vector1 = np.delete(columna_eliminada, [indice_maxa, indice_mina])
            vector2 = np.delete(fila_eliminada, [indice_maxb, indice_minb])
            vector3 = np.delete(columna_eliminada, [indice_maxa, indice_mina])
            maximos_por_posicion = np.maximum.reduce([vector1, vector2, vector3])

            p = (2 * l)
            nuevo_nn = np.array(maximos_por_posicion ** (p))
            matriz_uni = np.delete(np.delete(matriz_uni, indice_max, axis=0), indice_max, axis=1)
            matriz_n = np.zeros((len(matriz_uni) + 1, len(matriz_uni) + 1))
            matriz_n[1:, 1:] = matriz_uni
            matriz_n[0, 1:] = nuevo_nn
            matriz_n[1:, 0] = nuevo_nn
            
            
            if matriz_uni.shape[0] > 2 and matriz_uni.shape[1] > 2:
               matriz_sin_maximos = matriz_n[1:, 1:]  # Tomar la submatriz sin los valores máximos
               maximo = np.amax(matriz_sin_maximos)  # Calcular el máximo valor de similitud de la submatriz
            else:
               maximo = np.amax(matriz_n)  # T

            
            
            
            matriz_uni = matriz_n

            nueva_matriz = matriz_uni
            matriz_simetrica = np.array(nueva_matriz)
            maximo = np.amax(matriz_simetrica)

            nombres = [f'{etiquetasq[i]}' for i in range(matriz_simetrica.shape[0])]

            dfss = pd.DataFrame(matriz_simetrica, index=nombres, columns=nombres)

            w2 = matriz_simetrica.shape[0]
            q2 = matriz_simetrica.shape[1]

            for i in range(w2):
                for j in range(q2):
                    indice_max = np.unravel_index(np.argmax(matriz_simetrica), matriz_simetrica.shape)
                    fila_eliminada = np.ravel(matriz_simetrica[0:, indice_max[0]])
                    nuevo_encabezado = np.delete(nombres, indice_max, axis=0)

                    vector1 = np.array(nombres)
                    vector2 = np.array(nuevo_encabezado)

                    nuevo_vector = np.setdiff1d(vector1, vector2)
                    resultado_vector = ','.join(nuevo_vector)
                    vector3 = f"D{l}"
                    nuevo_vector_final = np.insert(nuevo_encabezado, 0, vector3)
                    
                    
                    
                    
            etiquetas = nombres
            etiquetasq = [etiqueta for etiqueta in etiquetas if etiqueta not in (nuevo_vector)]
            etiquetasq.insert(0, vector3)
            
            
            
             
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # Esta opción muestra la matriz completa
               
               datos_nivel = (k, maximo, nuevo_vector, vector3, dfss)
               contenedor_datos_niveles.append(datos_nivel)
               
               
            resultado_text.update_idletasks()
              
    

            resultado_text.see(tk.END)  # Desplazar hacia abajo para mostrar los resultados más recientes
            # Verificar si la matriz es toda cero después de las operaciones
            if np.all(matriz_uni == 0):
                break
   
   ### creacion del dendrograma  
    
    def dendograma_invertido(m):
        nueva_matriz1 = np.fill_diagonal(m, 1)
        # DENDOGRAMA INVERTIDO
        similaridad = hierarchy.distance.pdist(matriz)
        enlaces = hierarchy.linkage(similaridad, method='complete', metric='euclidean')
        dendrogram = hierarchy.dendrogram(enlaces, labels=column_labels, orientation='bottom')
        # Muestra los valores entre las parejas de variables
        for i, d, c in zip(dendrogram['icoord'], dendrogram['dcoord'], dendrogram['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            plt.plot(x, y)
            
        # Configura los ejes y muestra el gráfico
        plt.title('Dendrograma')
        plt.xlabel('Índices de muestra')
        plt.ylabel('SIMILARIDAD')
        plt.savefig('Dendo.pdf')
        plt.show()
        data_sim.to_excel('values.xlsx', index=False)
        dfn.to_excel('similaridad.xlsx', index=True , header=True)

    dendograma_invertido(matriz)

 
    


def ordenar_nodos(datos):
    # Limpiar contenedores
    contenedor_valor.clear()
    resultados.clear()
    contendor_diccionario.clear()
    contenedor_f.clear()
    contenedor_sumaI.clear()
    contenedor_rkk.clear()
    contenedor_skk.clear()
    fi.clear()
    Ii.clear()
    contenedor_v.clear()
   

    dfnuevo = pd.DataFrame(datos)
    v_1 = (dfnuevo[0] + "," + dfnuevo[1])
    v_2 = (dfnuevo[1] + "," + dfnuevo[0])
    a = np.array(v_1)
    b = np.array(v_2)
    c = np.array(dfnuevo[2])

    orden = np.argsort(c)[::-1]

    a_ordenada = a[orden]
    b_ordenada = b[orden]
    c_ordenada = c[orden]

    df_ordenados = pd.DataFrame({'Var1': a_ordenada, 'Var2': b_ordenada, 'Valor': c_ordenada})
    df_ordenadoc = pd.DataFrame({'Var1': a_ordenada, 'Var2': b_ordenada})
    df_ordenadov = pd.DataFrame({'Valor1': c_ordenada, 'Valor2': c_ordenada})

    df_combined1 = pd.DataFrame(df_ordenadoc.values.reshape(-1), columns=['Variables'])
    df_combined2 = pd.DataFrame(df_ordenadov.values.reshape(-1), columns=['Valor'])
    df_concatenado = pd.concat([df_combined1, df_combined2], axis=1)

    contenedor_valor.append(df_combined2)

    c_ordenado = df_concatenado['Valor'].values 

    c_orden = pd.Series(c_ordenado)

    grupos = c_orden.groupby(c_orden).groups
    num = 0
    for i, grupo in enumerate(grupos, 1):
        yy.append(grupo)
    xx.append(grupos)

    diccionario = xx[0]
    contenedor_valores = []
    contendor_diccionario.append(diccionario)

    dados = contendor_diccionario[0]
    w = 0
    for chave, valor in dados.items():
        w = w + 1
        lista = valor
        indice = pd.Index(valor)
        valores = indice.values.tolist()
        contenedor_valores.append(valores)

    variables = contenedor_valores

    def tabla_resolucion_nodos(variables, contenedor):
        
        v_alpha_k = 0
        s_k = len(contenedor)
        suma_total = 0
        mk = len(variables)
        for i in range(mk):
            contar = count(variables[i])
            resultados.append(contar)
            suma_total += contar
            f = (count(variables[i]) - 1)
            contenedor_sumaI.append(suma_total)
            contenedor_f.append(f)
        resultado_text.delete(1.0, tk.END)  # Limpiar el texto antes de agregar nuevos resultados
        resultado_text.insert(tk.END, " vector de nodos significativos\n")
        fila_invertida = contenedor_sumaI[::-1]
        columna_invertida = contenedor_f[::-1]
        s_k = (len((contenedor)) - 1)
        con_al = []
        contt=0
        
        for i in range(len(contenedor_f)):
            fi.append(columna_invertida[i])
            Ii.append(fila_invertida[i])
           
            rk = i + 1
            sk = s_k - i
            
            contenedor_rkk.append(rk)
            contenedor_skk.append(sk)
            card = (sum(Ii) - rk * ((rk + 1) / 2) - sum(fi))
            s_beta_k = round((card - (0.5 * sk * rk)) / math.sqrt((sk * rk * (sk + rk + 1)) / 12), 5)
            con_al.append(s_beta_k)
            valoresv = con_al
            resultados_v = []
            contenedor_vector = []
            for i in range(1, len(valoresv)):
                resta = valoresv[i] - valoresv[i - 1]
                contenedor_vector.append(resta)
            contenedor_vector.insert(0, valoresv[0])
            for j in range(len(contenedor_vector)):
                v = round(contenedor_vector[j], 3)
            contenedor_v.append(v)
            resultado_text.insert(tk.END, f"Card() [{i}] : {card}, S(Ω,k) [{i}]: {s_beta_k}, V(Ω,k)[{i}]: {v}\n")
        resultado_text.update_idletasks()

    tabla_resolucion_nodos(variables, contenedor_valor[0])


def grafica_omega(v):
    n = len(v)
    x = list(range(0, n))
    y = v
    nuevos_y = [valor if valor >= 0.0 else 0 for valor in v]  # Valores menores que 0.30 se establecen en 0
    colores = ['red' if valor > 0.0 else 'black' for valor in v]  # Lista de colores

    plt.plot(x, y, 'o-', color='black', alpha=0.5)  # Línea que une los puntos (negro opaco)

    for i in range(len(x)):
        if colores[i] == 'black':
            plt.scatter(x[i], y[i], c='black', label='Valores de V(Ω,k)', alpha=0.5)  # Puntos negros
            plt.text(x[i], y[i], str(y[i]), ha='center', va='bottom', color='black')  # Etiqueta para valores de V(Ω,k)
        else:
            plt.scatter(x[i], nuevos_y[i], c='red', label='Nodos significativos', alpha=0.5)  # Puntos rojos
            plt.text(x[i], nuevos_y[i]+0.02, str(y[i]), ha='center', va='bottom', color='red')  # Etiqueta para nodos significativos

    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Índice de k')
    plt.ylabel('Valor de V(Ω,k)')
    plt.title('Gráfico de V(Ω,k)')
    plt.tight_layout()
    plt.savefig('nodos.pdf')
    plt.show()

def seleccionar_archivo():
    archivo_excel = filedialog.askopenfilename(filetypes=[("Archivos Excel", "*.xlsx")])
    if archivo_excel:
        try:
            global base
            base = pd.read_excel(archivo_excel)
            # Limpia el cuadro de texto antes de cargar nuevos datos
            resultado_text.delete(1.0, tk.END)
            # Mostrar las variables en casillas de selección
            mostrar_variables()
        except Exception as e:
            resultado_text.delete(1.0, tk.END)
            resultado_text.insert(tk.END, f"Error al procesar: {str(e)}")
            
def salir():
    ventana.quit()            

def mostrar_variables():
    if len(base.columns) > 10:
        ventana_variables = tk.Toplevel()
        ventana_variables.title("Seleccionar Variables")

        frame = ttk.Frame(ventana_variables)
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas = tk.Canvas(frame, height=200)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        var_selection = []

        for i, variable in enumerate(base.columns):
            # Filtrar columnas sin nombre (como "Unnamed")
            if variable.startswith("Unnamed"):
                continue
            
            var_var = tk.BooleanVar()
            var_checkbutton = tk.Checkbutton(scrollable_frame, text=variable, variable=var_var)
            var_checkbutton.grid(row=i, column=0, sticky="w")
            var_selection.append((variable, var_var))

        # Botón para seleccionar todas las variables
        var_select_all_var = tk.BooleanVar()
        checkbutton_select_all = tk.Checkbutton(ventana_variables, text="Seleccionar Todas", variable=var_select_all_var, command=lambda: select_all_variables(var_selection, var_select_all_var))
        checkbutton_select_all.grid(row=i+1, column=0, sticky="w")

        boton_calcular = tk.Button(ventana_variables, text="Calcular Similaridad", command=lambda: calcular_similaridad_desde_seleccion(var_selection))
        boton_calcular.grid(row=i+2, column=0, pady=10)
    else:
        ventana_variables = tk.Toplevel()
        ventana_variables.title("Seleccionar Variables")

        var_selection = []

        for variable in base.columns:
            # Filtrar columnas sin nombre (como "Unnamed")
            if variable.startswith("Unnamed"):
                continue
            
            var_var = tk.BooleanVar()
            var_checkbutton = tk.Checkbutton(ventana_variables, text=variable, variable=var_var)
            var_checkbutton.pack()
            var_selection.append((variable, var_var))

        boton_calcular = tk.Button(ventana_variables, text="Calcular Similaridad", command=lambda: calcular_similaridad_desde_seleccion(var_selection))
        boton_calcular.pack()


def select_all_variables(var_selection, var_select_all_var):
    select_all_state = var_select_all_var.get()
    for _, var_var in var_selection:
        var_var.set(select_all_state)

def mostrar_ayuda():
    # Abre el archivo de ayuda con el programa predeterminado del sistema
    archivo_ayuda = "Manual.pdf"  # Nombre del archivo de ayuda
    if os.path.exists(archivo_ayuda):  # Verifica si el archivo existe
        os.system(f'start {archivo_ayuda}')  # Abre el archivo con el programa predeterminado
    else:
        resultado_text.delete(1.0, tk.END)
        resultado_text.insert(tk.END, "El archivo de ayuda no está disponible.\n")


def calcular_similaridad_desde_seleccion(var_selection):
    global variables_seleccionadas
    variables_seleccionadas = [variable for variable, var_var in var_selection if var_var.get()]

    if len(variables_seleccionadas) >= 2:
        base_seleccionada = base[variables_seleccionadas]
        calcular_similitud(base_seleccionada)
        
    else:
        resultado_text.delete(1.0, tk.END)
        resultado_text.insert(tk.END, "Seleccione al menos dos variables para calcular la similaridad.\n")
        

import tempfile
import os

def mostrar_matriz_completa():
    # Abrir el archivo CSV con Excel
    os.system('start excel.exe similaridad.xlsx')  # Esto abrirá el archivo CSV con Excel en Windows

def mostrar_niveles_similaridad():
    ventana_matrices = tk.Toplevel()
    ventana_matrices.title("Matrices de Niveles")

    cuadro_texto = tk.Text(ventana_matrices, height=500, width=500)
    cuadro_texto.pack()

    if contenedor_datos_niveles:
        for nivel, maximos, nuevo_vector, vector3, dfss in contenedor_datos_niveles:
            if nivel == 0 :
                
                cuadro_texto.insert(tk.END, "---------------------Niveles de Similaridad----------------------------\n")
                
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    cuadro_texto.insert(tk.END, str(vector3) + "\n")
                    
            else:
                # Verificar si la matriz es toda cero
                if np.all(dfss.values == 0):
                    # Si es la última matriz, no la imprimimos
                    if nivel == len(contenedor_datos_niveles) - 1:
                        break
                    else:
                        continue
                
                cuadro_texto.insert(tk.END, "-----------------------------------------------------------------------------------\n")
                cuadro_texto.insert(tk.END, f" MATRIZ DE NIVEL {nivel}\n")
                
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    cuadro_texto.insert(tk.END, f"{dfss}\n\n")
                cuadro_texto.insert(tk.END, f"VARIABLES {nuevo_vector} = {vector3}\n")
                cuadro_texto.insert(tk.END, f"VARIABLES {nuevo_vector}, VALOR NIVEL : {maximos}\n\n")
                
                # Imprimir variables contenidas en el vector actual
                variables_vector_actual = [var for var in nuevo_vector if not var.startswith(("V", "A", "D"))]
                cuadro_texto.insert(tk.END, f"{vector3} contiene las variables: {', '.join(variables_vector_actual)}\n\n")

                # Imprimir variables contenidas en los vectores anteriores (si existen)
                variables_previas = [var for var in nuevo_vector if var.startswith(("V", "A", "D"))]
                for var_previa in variables_previas:
                    variables_previa_nombres = [var for var in nuevo_vector if var.startswith(var_previa + ",")]
                    cuadro_texto.insert(tk.END, f"{var_previa} contiene las variables: {', '.join(variables_previa_nombres)}\n")

                
                
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    cuadro_texto.insert(tk.END, str(vector3) + "\n\n")
    else:
        cuadro_texto.insert(tk.END, "No se ha calculado la similitud aún.\n")


def calcular_nodos_significativos():
    if contenedor_lista_matriz:
        ordenar_nodos(contenedor_lista_matriz)
        grafica_omega(contenedor_v)
    else:
        resultado_text.delete(1.0, tk.END)
        resultado_text.insert(tk.END, "Primero calcula la similaridad.\n")

def borrar_valores():
    resultado_text.delete(1.0, tk.END)
    contenedor_valor.clear()
    contenedor_v.clear()
    contenedor_valor.clear()
   # Restablecer la lista de valores de nodos
    contenedor_de_la_matriz.clear()  # Restablecer la lista de matrices de similaridad
    contendor_diccionario.clear()  # Restablecer el diccionario de nodos
    contenedor_f.clear()
    contenedor_sumaI.clear()
    contenedor_rkk.clear()
    contenedor_skk.clear()
    contenedor_lista_matriz.clear()
    fi.clear()
    Ii.clear()
    yy.clear()
    xx.clear()
    contenedor_datos_niveles.clear()
    
    

ventana = tk.Tk()
ventana.title("Calcular Similaridad y Nodos Significativos")

menubar = tk.Menu(ventana)
archivo_menu = tk.Menu(menubar, tearoff=0)
archivo_menu.add_command(label="Seleccionar Archivo Excel", command=seleccionar_archivo)
archivo_menu.add_command(label=" Nodos Significativos", command=calcular_nodos_significativos)
archivo_menu.add_command(label="Limpiar", command=borrar_valores)
archivo_menu.add_command(label="Salir", command=salir)
menubar.add_cascade(label="Similaridad Lerman", menu=archivo_menu)

ver_menu = tk.Menu(menubar, tearoff=0)
ver_menu.add_command(label="Ver Matriz Completa", command=mostrar_matriz_completa)
ver_menu.add_command(label="Ver Niveles de Similaridad", command=mostrar_niveles_similaridad)
menubar.add_cascade(label="Ver", menu=ver_menu)

Ayuda_menu = tk.Menu(menubar, tearoff=0)
Ayuda_menu.add_command(label="Ayuda", command=mostrar_ayuda)
menubar.add_cascade(label="Ayuda", menu=Ayuda_menu)

ventana.config(menu=menubar)

resultado_text = tk.Text(ventana, height=250, width=200)
resultado_text.pack(pady=100)

ventana.mainloop()
