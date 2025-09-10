# Brazo Visual Pick & Place

Este proyecto implementa un sistema de control para un brazo robótico tipo pick & place, con integración de visión artificial y macros automatizadas. El objetivo es permitir la clasificación y manipulación de objetos en base a la detección por cámara y la ejecución de rutinas de movimiento.

## Descripción General

El sistema está compuesto por los siguientes módulos principales:

- **app.py**: Punto de entrada principal. Orquesta la carga de configuración, inicialización del robot, visión y la máquina de estados (FSM) para el ciclo pick & place.
- **robot/**: Drivers para distintos tipos de brazos robóticos (simulado, G-code, Dobot, etc.). Implementan la interfaz común `RobotArm`.
- **macros/**: Rutinas de alto nivel para movimientos típicos (HOME, PICK, PLACE, STOP_ALL).
- **planner/**: Implementa la máquina de estados finitos (FSM) que gestiona el ciclo de trabajo y la lógica de despacho.
- **vision/**: Módulos para cámara, clasificación de imágenes, detección de presencia y normalización de ROI.
- **ui/**: Interfaz gráfica para control manual y monitoreo.
- **configs/**: Archivos YAML de configuración para el robot, visión, poses y despacho.

## Instalación

1. **Clona el repositorio** y entra en la carpeta del proyecto.
2. **Activa el entorno virtual** (ya incluido en `env/`):
   ```powershell
   .\env\Scripts\Activate.ps1
   ```
3. **Instala las dependencias** (si es necesario):
   ```powershell
   pip install -r requirements.txt
   ```
   *Dependencias principales: PyYAML, (otros según drivers y visión)*

## Uso

1. Configura los archivos YAML en la carpeta `configs/` según tu hardware y necesidades.
2. Ejecuta el sistema principal:
   ```powershell
   python app.py
   ```
3. Usa la interfaz gráfica (`ui/gui.py`) si deseas control manual.

## Ejemplo de Flujo

- El sistema inicia y carga la configuración.
- Se conecta al brazo robótico y lo lleva a la posición HOME.
- Espera la presencia de un objeto detectado por la cámara.
- Captura imágenes, las clasifica y decide el destino del objeto.
- Ejecuta las macros de PICK y PLACE según la clasificación.
- Vuelve a HOME y repite el ciclo.

## Estructura de Carpetas

- `app.py` — Orquestador principal
- `robot/` — Drivers de brazo
- `macros/` — Rutinas de movimiento
- `planner/` — FSM y lógica de despacho
- `vision/` — Visión artificial
- `ui/` — Interfaz gráfica
- `configs/` — Configuración YAML
- `tests/` — Pruebas unitarias

## Personalización

- Puedes agregar nuevos drivers de robot en `robot/` siguiendo la interfaz `RobotArm`.
- Modifica las macros en `macros/sequences.py` para rutinas personalizadas.
- Ajusta la lógica de la FSM en `planner/fsm.py` para nuevos ciclos o políticas.

## Créditos

Desarrollado por Tomas Bond (Fundación UADE).

---

Para dudas o mejoras, contacta al autor o abre un issue en el repositorio.
