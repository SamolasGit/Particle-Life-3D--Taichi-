# Particle Life 3D

Particle Life 3D is a **GPU-accelerated 3D particle simulation** built with **Taichi**, combining multiple particle-based interaction models into a single **emergent system**.

Particles are divided into distinct **species** that **attract or repel** each other according to a randomized interaction matrix. When particles come into close contact, they can **react**, transforming into new species and dynamically altering the system’s behavior over time.

The simulation runs entirely on the **GPU** and features a **real-time 3D camera** with interactive controls, allowing live exploration and tuning of simulation parameters.

---

## Requirements

* **Python 3.9+**
* **Taichi** (GPU backend recommended: CUDA / Vulkan / Metal)

### Python Dependencies

```txt
taichi
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Simulation

```bash
python main.py
```

(Replace `main.py` with your script filename.)

---

## Controls

### Camera Controls

* **Right Mouse Button + Mouse** → Look around
* **W / A / S / D** → Move forward / left / backward / right
* **Q / E** → Move down / up
* **Shift** → Increase camera speed

### Simulation Controls

* **Spacebar** → Pause / Resume simulation

---

## GUI Parameters

The following parameters can be adjusted in real time:

| Parameter | Description                 |
| --------- | --------------------------- |
| SPACE     | Size of the simulation cube |
| BETA      | Core repulsion threshold    |
| RADIUS    | Interaction radius          |
| TIME STEP | Simulation speed            |

---

## Core Concepts

### Species Interaction

Each particle belongs to a **species**. Every species pair has an attraction value:

* **Positive values** → Attraction
* **Negative values** → Repulsion

These values are stored in a **species interaction matrix** and randomized at startup.

---

### Reactions

When two particles are closer than a fixed **reaction distance**, they may react:

```
A + B → C + D
```

* Reactions are randomly generated at startup
* Both particles can change species
* Particle colors update automatically
* The total number of reactions is tracked internally

This allows **artificial chemistry–like behavior** to emerge.

---

## Force Model

The interaction force depends on distance:

* Strong repulsion at very close range
* Attraction or repulsion within a fixed radius
* No interaction beyond the radius

This non-linear force model produces clustering, separation, dynamic pattern formation, and long-term emergent structures.

---

## Performance Notes

⚠️ The simulation uses an **O(N²)** interaction model.

Recommendations:

* Keep `NUM_PARTICLES` relatively low (GPU-dependent)
* Gradually increase particle count
* Consider spatial partitioning for optimization

---

## Customization Ideas

* Increase particle count for richer dynamics
* Manually design reaction rules
* Add particle reproduction or extinction
* Save reaction history to disk
* Add sound effects triggered by reactions
* Replace random attraction with learned matrices
* Turn it into a scientific experiment or visualization tool

---

## Inspiration

This project is inspired by:

- Particle Life simulations
- Artificial chemistry systems
- Emergent behavior and complexity theory
- [Digital Genius — Particle Life](https://youtu.be/4vk7YvBYpOs?si=v_zpsa_sHmO_cVhU)
- [How Life Arises from Simplicity](https://youtu.be/p4YirERTVF0?si=rWTHM42dkZqJ97kU)
- [Particle Life with Reactions](https://youtu.be/JmCN_4jTwf8?si=HMYes6S5ooMqj41J)
- [3D Particle Life Adaptation](https://youtu.be/ZTBwuU_zvxk?si=AJ5P2e6l_jXVivKR)

---

## License

MIT License

You are free to use, modify, and distribute this project for personal or commercial purposes.
