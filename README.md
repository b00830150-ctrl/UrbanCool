# 🏙️ UrbanCool

**UrbanCool** est une application **Streamlit** qui cartographie les **îlots de chaleur urbains (UHI)** à partir de données de température satellite/air, de couverture du sol et de population.  
Elle propose ensuite des **interventions de verdissement à faible coût** (plantations d’arbres, toits frais, toits végétalisés), priorisées selon **l’exposition de la population**.

---

## 🚀 Fonctionnalités principales

- Détection automatique des **îlots de chaleur urbains (UHI)** à partir d’un raster de température.  
- Intégration facultative :
  - Raster de **landcover** (couverture du sol)
  - Raster de **population**
  - Raster de **bâti / ratio de toits**
- Calcul de l’exposition populationnelle.
- Proposition d’interventions adaptées (arbres, toits frais, végétalisation).
- Classement des zones par **score de priorité**.
- Visualisation interactive sur **carte Folium**.
- Export des résultats en **GeoJSON** et **CSV**.

---

## 🧰 Installation

### 1️⃣ Cloner le projet
```bash
git clone https://github.com/ton-compte/UrbanCool.git
cd UrbanCool