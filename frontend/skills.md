# Frontend Skills For Churn Prediction App

## Purpose
- Build and maintain a production-friendly UI for churn prediction.
- Keep frontend code simple, testable, and API-driven.

## Core Skills
- Form design for structured customer data input.
- API integration with FastAPI backend endpoint `/predict`.
- Client-side validation for required fields and numeric ranges.
- Result rendering with clear labels: prediction class and probability.
- Error handling for API/network/server failures.

## Production Standards
- Keep JavaScript modular and readable.
- Avoid hardcoded backend host if served from same origin.
- Use semantic HTML and accessible labels.
- Use responsive CSS for desktop and mobile.
- Show loading states while waiting for inference responses.

## Contract With Backend
- Endpoint: `POST /predict`
- Request shape:
  - `model_type`: `"bayesian"` or `"grid"`
  - `records`: list with one or more customer objects
- Response shape:
  - `model_type`
  - `count`
  - `predictions`
  - `churn_probability`

## Development Workflow
1. Validate backend health using `/health`.
2. Submit a sample payload from the UI.
3. Confirm response mapping in UI cards/table.
4. Verify edge cases (missing fields, bad numbers).
5. Check mobile layout and browser console errors.

## Future Enhancements
- Batch upload CSV and preview.
- Authentication for protected inference APIs.
- Usage analytics for model requests.
- Feature flags for model version switching.
