# Agente de Conducción Autónoma con Q-Learning

Este proyecto implementa un agente de aprendizaje por refuerzo (Q-Learning) que aprende a conducir un vehículo autónomo en un entorno simulado de 3 carriles, evitando obstáculos.

El sistema incluye una visualización en tiempo real construida con `pygame` que muestra el proceso de aprendizaje, estadísticas de rendimiento y métricas clave.

## Estructura del Proyecto

*   `main.py`: Punto de entrada de la aplicación.
*   `requirements.txt`: Lista de dependencias del proyecto.
*   `src/`: Directorio con el código fuente.
    *   `agent.py`: Implementación del agente `QLearningAgent`.
    *   `env.py`: Entorno de simulación `LaneEnv`.
    *   `train.py`: Lógica de entrenamiento y gestión de episodios.
    *   `render.py`: Interfaz gráfica (UI) y visualización.
    *   `stats.py`: Gestión de estadísticas en vivo.

## Instalación

1.  **Clonar el repositorio** (si aplica) o descargar el código.
2.  **Crear un entorno virtual** (recomendado):
    ```bash
    python -m venv venv
    # En Windows:
    .\venv\Scripts\activate
    # En Mac/Linux:
    source venv/bin/activate
    ```
3.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

Para iniciar la simulación:

```bash
python main.py
```

### Controles en la Interfaz

*   **`T`**: Entrenar 50 episodios rápidamente (Fast-forward). Útil para acelerar el aprendizaje.
*   **`P`**: Pausar / Reanudar la reproducción automática ("Play Mode"). En modo Play, el agente actúa solo de forma voraz (sin exploración aleatoria) para demostrar lo aprendido.
*   **`R`**: Reiniciar el entorno manualmente.

## Visualización

La interfaz muestra:
*   **Simulación**: El coche (verde) esquivando obstáculos (círculos rojos) que bajan por los carriles.
*   **Estadísticas en vivo**:
    *   Número de episodio actual.
    *   Distancia recorrida en el último episodio.
    *   Media móvil de la distancia (últimos 20 episodios).
    *   Tasa de choques (Crash rate).
    *   Valor actual de Epsilon (probabilidad de exploración).
*   **Gráficas**: Evolución de la distancia, media móvil y decaimiento de epsilon.

## Cómo funciona

El agente utiliza **Q-Learning Tabular**.
*   **Estado**: Se define por el carril actual del coche y la distancia discretizada (bins) a los obstáculos más cercanos en cada uno de los 3 carriles.
*   **Acciones**: Izquierda, Mantenerse, Derecha.
*   **Recompensa**: +1 por cada paso sin chocar, -10 por chocar.
