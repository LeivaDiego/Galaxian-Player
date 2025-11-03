# Laboratorio 10: Interfaz del Proyecto Final

**Universidad del Valle de Guatemala**  
**Facultad de Ingeniería**  
**Departamento de Ciencias de la Computación**  
**Aprendizaje por Refuerzo**

## Autor

- Diego Leiva       -   21752

## Entorno y código base

### Objetivos y mecánicas del juego

Galaxian es un **shooter** 2D. El jugador controla una nave que se **mueve horizontalmente** y dispara a enemigos que atacan en **formaciones** y **picadas**.
**Objetivo:** maximizar el puntaje destruyendo enemigos y evitando colisiones y proyectiles.
**Fin de episodio:** al perder las vidas o por truncamiento del entorno.
**Puntaje:** suma de recompensas por enemigos destruidos durante el episodio.

![alt text](image.png)

### Definición del estado del entorno

En `ALE/Galaxian-v5` la **observación** es una imagen RGB `uint8` de forma aproximada `(210, 160, 3)` por paso.
El agente recibe este **frame crudo** y decide una acción; en este prerequisito no transformamos la observación. El script captura cada frame para generar el MP4.

### Acciones disponibles

Espacio **discreto**. Acciones típicas expuestas por ALE para Galaxian:

* `0: NOOP`
* `1: FIRE`
* `2: RIGHT`
* `3: LEFT`
* `4: RIGHTFIRE`
* `5: LEFTFIRE`