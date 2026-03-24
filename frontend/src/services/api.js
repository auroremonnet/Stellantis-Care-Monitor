/**
 * src/services/api.js — Stellantis CARE Monitor : Service API React
 * ──────────────────────────────────────────────────────────────────
 * Toutes les communications avec server.py passent par ici.
 * Le contrat API respecte exactement ce qu'attend App.js :
 *   api.sendFrame(frameData, temperature)   → { emotion, primary_emotion, annotated_image, temperature, ... }
 *   api.checkVLM()                          → { question }
 *   api.sendVLMResponse(response)           → { status, new_target }
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Wrapper fetch avec gestion d'erreur centralisée.
 * Lance une erreur explicite si le serveur est injoignable.
 */
async function apiFetch(endpoint, options = {}) {
  const url = `${API_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!res.ok) {
      throw new Error(`[API] ${endpoint} → HTTP ${res.status}`);
    }
    return res.json();
  } catch (err) {
    // Ne pas spammer la console à chaque frame si le serveur est down
    if (err.name !== 'TypeError') console.error(err.message);
    throw err;
  }
}

export const api = {
  /**
   * Envoie une frame JPEG (data URL base64) au pipeline d'analyse.
   *
   * @param {string} frameData   - "data:image/jpeg;base64,..."
   * @param {number} temperature - Température courante (référence)
   * @returns {{ emotion, primary_emotion, annotated_image, temperature, state, stats, climate_mode }}
   */
  sendFrame: (frameData, temperature) =>
    apiFetch('/analyze', {
      method: 'POST',
      body: JSON.stringify({ frame: frameData, temperature }),
    }),

  /**
   * Vérifie si le moteur VLM a généré une question à afficher.
   * Appelé par le frontend toutes les secondes (voir App.js).
   *
   * @returns {{ question: string | null }}
   */
  checkVLM: () =>
    apiFetch('/vlm/check'),

  /**
   * Transmet la réponse utilisateur (OUI / NON) à la question VLM.
   * Le backend ajuste la température cible en conséquence.
   *
   * @param {string} userResponse - "oui" ou "non"
   * @returns {{ status: string, new_target: number }}
   */
  sendVLMResponse: (userResponse) =>
    apiFetch('/vlm/response', {
      method: 'POST',
      body: JSON.stringify({ response: userResponse }),
    }),

  /**
   * Optionnel — utile pour afficher un indicateur de connexion dans l'UI.
   * @returns {{ status: string, engine_state: string, temperature: number }}
   */
  healthCheck: () =>
    apiFetch('/health'),
};