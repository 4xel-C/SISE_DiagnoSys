### Development Rules & Best Practices

#### General Guidelines

* All code comments, documentation, commit messages, and pull requests **must be written in English**.
* Follow clean code principles: readability, simplicity, and maintainability are mandatory.

#### Package & Environment Management

* **UV** must be used for Python package and dependency management.
* Dependency versions should be explicit and reproducible. (use `UV add 'package'` commands to add any package to the project).

#### Git Workflow

* Each new feature or fix **must be developed in its own branch**, created from `main`.
* Before requesting a merge:

  * Rebase or merge `main` into your branch to ensure it is up to date.
  * Ensure all tests pass and the application runs correctly.
* A **Pull Request (PR)** is required for every merge into `main`.
* **No direct commits to `main`** are allowed, `main` is thus deactivated pretected from direct merge.
* **Use 'Issues' to monitor any bug and implement a 'TODO' list for the team.

#### Code Review Policy

* **At least one code review is mandatory** before merging any PR into `main`.
* Reviewers must check:

  * Code quality and readability
  * Architecture and separation of concerns
  * Typing, documentation, and test coverage
* PRs should be small, focused, and easy to review.

#### Code Quality & Standards

* All Python functions and methods must:

  * Be fully **type-annotated**
  * Include clear **docstrings** explaining purpose, parameters, and return values
* Use **Object-Oriented Programming (OOP)** principles:

  * Proper encapsulation
  * Clear responsibilities per class
  * No unnecessary global state
* Maintain a **clean and intentional project structure**:

  * No “free helper files” without a clear and documented purpose
  * Each module must have a well-defined responsibility

#### Architecture Rules

* **Strict separation of concerns is required**:

  * Core/business logic must live in the **services or domain layer**
  * UI, routing, or API layers must remain thin
* The UI/routing layer:

  * Must not contain calculations or business rules
  * Must not implement complex logic or data transformations
  * Should only orchestrate calls to services and format input/output
  
#### Logger usasge
3. Utilisation dans chaque fichier

En haut de CHAQUE fichier :
import logging
logger = logging.getLogger(__name__)

Dans le code :
```python
# app/routes/query.py
import logging
logger = logging.getLogger(__name__)  # 'app.routes.query'

@bp.route('/query')
def query():
    logger.info("Requête reçue")

    try:
        result = process()
        logger.debug(f"Résultat: {result}")
        return result
    except ValueError as e:
        logger.warning(f"Validation échouée: {e}")
    except Exception as e:
        logger.exception(f"Erreur: {e}")  # Avec stack trace
```

4. Niveaux de log

logger.debug("Détails techniques")      # Développement
logger.info("Opération normale")        # Confirmation
logger.warning("Situation anormale")    # Attention
logger.error("Erreur fonctionnelle")    # Échec
logger.exception("Erreur avec trace")   # Dans except
logger.critical("Erreur fatale")        # Application KO

#### Final Principle

> If a piece of code does more than one thing, it probably belongs somewhere else.







