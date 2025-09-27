# AI Agent Implementation Prompts

This directory contains comprehensive implementation prompts for building the AI Agent Application based on the architecture plan in `@ai-agent-architecture-plan.md`.

## 📋 **Implementation Phases Overview**

### **Phase 1: Foundation Setup**
- [Phase 1.1 & 1.2: Project Structure & Domain Models](./phase1-foundation-setup.md)
- **Goal**: Complete project scaffold and domain layer implementation
- **Duration**: 1-2 weeks
- **Dependencies**: None

### **Phase 2: Infrastructure Layer**
- [Phase 2.1 & 2.2: Configuration & Repository Pattern](./phase2-infrastructure-layer.md)
- **Goal**: Configuration management and data persistence layer
- **Duration**: 1-2 weeks
- **Dependencies**: Phase 1 complete

### **Phase 3: Resilience Layer**
- [Phase 3.1 & 3.2: Retry & Circuit Breakers](./phase3-resilience-layer.md)
- **Goal**: External service resilience patterns and health checking
- **Duration**: 1 week
- **Dependencies**: Phase 2 complete

### **Phase 4: API Layer**
- [Phase 4.1 & 4.2: REST API & WebSocket](./phase4-api-layer.md)
- **Goal**: Complete API implementation with real-time communication
- **Duration**: 1-2 weeks
- **Dependencies**: Phase 3 complete

### **Phase 5: External Services**
- [Phase 5.1 & 5.2: LLM & MCP Integration](./phase5-external-services.md)
- **Goal**: LLM provider and MCP server integration
- **Duration**: 1-2 weeks
- **Dependencies**: Phase 4 complete

### **Phase 6: Production Features**
- [Phase 6.1 & 6.2: Security & Observability](./phase6-production-features.md)
- **Goal**: Enterprise security, secret management, and monitoring
- **Duration**: 1-2 weeks
- **Dependencies**: Phase 5 complete

### **Phase 7: Deployment & DevOps**
- [Phase 7.1 & 7.2: Docker, K8s & CI/CD](./phase7-deployment-devops.md)
- **Goal**: Production deployment and operations automation
- **Duration**: 1 week
- **Dependencies**: Phase 6 complete

## 🎯 **How to Use These Prompts**

### **Sequential Implementation Strategy**

1. **Start with Phase 1** - Foundation and domain models must be completed first
2. **Progress through phases sequentially** - Each phase builds on the previous ones
3. **Test thoroughly at each phase** - Ensure functionality before moving forward
4. **Reference the architecture plan** - Always include `@ai-agent-architecture-plan.md` as context

### **Prompt Usage Guidelines**

#### **For Each Phase:**
1. **Include the architecture plan** in your conversation context
2. **Use the specific prompt** from the corresponding phase file
3. **Reference exact sections** mentioned in the prompt
4. **Follow implementation specifications** precisely
5. **Test the implementation** before proceeding to the next phase

#### **Example Usage:**
```
I want to implement Phase 1.1 from the AI Agent implementation plan.

Please use the prompt from @phase1-foundation-setup.md Phase 1.1,
and reference the specifications in @ai-agent-architecture-plan.md.

[Copy the exact prompt from Phase 1.1 here]
```

### **Quality Assurance**

#### **For Each Implementation:**
- ✅ **Code matches architecture specifications exactly**
- ✅ **All dependencies and versions are correct**
- ✅ **Proper error handling and logging included**
- ✅ **Type hints and documentation complete**
- ✅ **Tests written and passing**
- ✅ **Integration points verified**

## 📁 **File Organization**

```
prompts/implementation/
├── README.md                          # This overview file
├── phase1-foundation-setup.md          # Project setup & domain models
├── phase2-infrastructure-layer.md      # Configuration & repositories
├── phase3-resilience-layer.md         # Retry logic & circuit breakers
├── phase4-api-layer.md                 # REST API & WebSocket
├── phase5-external-services.md        # LLM & MCP integration
├── phase6-production-features.md      # Security & observability
└── phase7-deployment-devops.md        # Docker, K8s & CI/CD
```

## 🔗 **Key Architecture References**

### **Always Reference These Sections:**
- **Section 1**: Domain Model - Core entities and relationships
- **Section 2**: Project Structure - Complete folder hierarchy
- **Section 3**: Technology Stack - Exact versions and dependencies
- **Section 4**: Configuration Strategy - Environment and settings
- **Section 5**: Infrastructure Strategy - Persistence and deployment
- **Section 6**: API Specification - Complete REST API design
- **Section 7**: Resilience Architecture - Retry and circuit breaker patterns
- **Section 8**: Implementation Roadmap - Development phases

## ⚡ **Quick Start Guide**

### **Step 1: Environment Setup**
1. Use Phase 1.1 prompt to create project structure
2. Verify all dependencies from Section 3 are installed
3. Test development environment setup

### **Step 2: Domain Implementation**
1. Use Phase 1.2 prompt to implement domain models
2. Copy exact code from Section 1 of architecture plan
3. Verify all entities and exceptions work correctly

### **Step 3: Continue Sequential Implementation**
1. Follow phases 2-7 in order
2. Test each phase thoroughly before proceeding
3. Reference architecture plan sections as specified

## 🎯 **Success Criteria**

### **Phase Completion Checklist:**
- [ ] All code matches architecture specifications
- [ ] All tests pass with good coverage
- [ ] Documentation is complete and accurate
- [ ] Integration points are verified
- [ ] Performance meets requirements
- [ ] Security requirements are implemented
- [ ] Ready for next phase dependencies

### **Final Implementation Goals:**
- ✅ **Production-ready AI agent application**
- ✅ **Complete resilience patterns implemented**
- ✅ **Full CRUD API with real-time capabilities**
- ✅ **Multi-LLM provider support with fallback**
- ✅ **MCP server integration with dynamic tools**
- ✅ **Enterprise security and secret management**
- ✅ **Comprehensive observability and monitoring**
- ✅ **Production deployment with CI/CD**

## 📞 **Support and Troubleshooting**

### **If Implementation Issues Arise:**
1. **Double-check architecture plan references** - Ensure you're following exact specifications
2. **Verify dependency versions** - Use exact versions from Section 3
3. **Review error handling patterns** - Follow resilience patterns from Section 7
4. **Check integration points** - Ensure proper dependency injection and interfaces
5. **Validate configuration** - Use validation patterns from Section 4

### **Common Implementation Patterns:**
- **Always use async/await** for I/O operations
- **Include proper error handling** with custom exceptions
- **Apply resilience patterns** to all external service calls
- **Use dependency injection** for testability
- **Follow type safety** with Pydantic v2 throughout
- **Include comprehensive logging** with correlation IDs

---

**Ready to build a production-ready AI agent application!** 🚀

Start with Phase 1 and work through each phase systematically, referencing the architecture plan at every step.
