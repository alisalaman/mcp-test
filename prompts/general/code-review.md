Please conduct a thorough code review of the provided codebase, focusing on the following areas:

1. Simplicity & Readability
- Review all functions and classes for unnecessary complexity
- Identify opportunities to simplify conditional logic and reduce nesting
- Look for overly verbose implementations that could be more concise
- Check for redundant code patterns that could be abstracted
- Ensure code follows the principle of "simple is better than complex"

2. Package Dependencies & Versions
- Verify all dependencies are using the latest stable versions
- Check for any deprecated packages or security vulnerabilities
- Ensure version constraints are appropriate (not too restrictive or too permissive)
- Look for unused dependencies that could be removed
- Verify compatibility between different package versions

3. Code Conciseness
- Remove verbose implementations where simpler alternatives exist
- Look for over-engineering in simple operations
- Check for unnecessary intermediate variables or complex expressions
- Identify code that could benefit from more idiomatic patterns
- Ensure each line of code serves a clear purpose

4. Debugging Artifacts
- Remove any print() statements or temporary logging added during debugging
- Clean up commented-out code blocks
- Remove TODO comments that have been addressed
- Look for development-only code that should be removed or properly configured
- Ensure no sensitive information is exposed in logs or comments

5. Conditional Logic Patterns
- Replace complex if-then-else chains with more readable patterns:
 - Use early returns to reduce nesting
 - Consider guard clauses for validation
 - Replace multiple conditions with dictionary lookups or match statements
 - Use list comprehensions or generator expressions where appropriate
 - Look for opportunities to use modern language features (e.g., walrus operator, pattern matching)
 - Ensure switch/select statements are used appropriately

6. Code Quality & Standards
- Run comprehensive linting checks (language-specific linters)
- Ensure all formatting follows project standards
- Verify all type hints/annotations are correct and complete
- Check for unused imports, variables, or functions
- Ensure consistent error handling patterns
- Verify naming conventions are followed consistently

7. Language-Specific Naming Conventions
- Review all variable names, function names, and class names
- Ensure they follow the project's language conventions:
 - English: Use British English spelling where specified
 - Consistency: Maintain consistent naming patterns throughout
 - Clarity: Names should clearly indicate purpose and scope
 - Conventions: Follow language-specific naming conventions (camelCase, snake_case, etc.)

8. Performance & Scalability
- Look for unnecessary database queries or API calls
- Check for inefficient loops or data processing
- Verify that caching is used appropriately
- Review memory usage patterns and potential leaks
- Identify bottlenecks in critical paths
- Ensure algorithms have appropriate time/space complexity
- Check for opportunities to use async/parallel processing
- Verify resource cleanup and disposal patterns

9. Security
- Verify input validation and sanitisation
- Check authentication and authorisation patterns
- Review error messages for information leakage
- Ensure proper handling of sensitive data
- Check for SQL injection, XSS, and other common vulnerabilities
- Verify secure communication protocols
- Review access control and permission checks
- Ensure secrets and credentials are handled securely

10. Testing Coverage & Quality
- Ensure comprehensive test coverage for core use cases
- Verify tests for significantly important edge cases
- Check that tests are maintainable and not brittle
- Ensure proper test isolation and independence
- Verify both positive and negative test scenarios
- Check for integration tests where appropriate
- Ensure tests provide clear failure messages
- Verify test data setup and teardown

11. Architecture & Design
- Review adherence to SOLID principles
- Check for proper separation of concerns
- Verify dependency injection patterns
- Ensure appropriate abstraction levels
- Check for circular dependencies
- Review design patterns usage
- Verify configuration management

12. Error Handling & Resilience
- Ensure comprehensive error handling
- Check for proper exception propagation
- Verify graceful degradation patterns
- Review retry mechanisms and circuit breakers
- Check for proper logging and monitoring
- Ensure user-friendly error messages

13. Documentation & Comments
- Ensure all public APIs have proper documentation
- Check that complex logic is well-commented
- Verify that type hints/annotations are comprehensive
- Remove any outdated or incorrect comments
- Ensure README and setup instructions are current
- Verify API documentation is complete and accurate

14. Maintainability
- Check for code duplication that could be refactored
- Ensure consistent coding patterns across the codebase
- Verify that code is self-documenting where possible
- Check for proper module/package organisation
- Ensure configuration is externalised appropriately

15. Specific Language/Framework Considerations
- Python: Follow PEP 8, use type hints, leverage modern Python features
- JavaScript/TypeScript: Use modern ES6+ features, proper async/await patterns

16. Review Output Format
- For each issue found, provide:
- Priority level (Critical/High/Medium/Low)
- Category (Performance/Security/Readability/etc.)
- Specific file and line numbers where applicable
- Before/after code examples for suggested improvements
- Rationale for each recommendation
- Impact assessment (performance, security, maintainability)

17. Focus Areas by Priority
- Critical: Security vulnerabilities, data corruption risks, critical performance issues
- High: Major architectural problems, significant performance bottlenecks, missing error handling
- Medium: Code quality issues, maintainability concerns, test coverage gaps
- Low: Style inconsistencies, minor optimisations, documentation improvements

Please provide specific, actionable recommendations that will improve the codebase's quality, security, performance, and maintainability while ensuring it remains functionally correct.
