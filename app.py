import streamlit as st
from model import RelationshipChatClassifier

def main():
    st.set_page_config(
        page_title="Relationship Chat Analyzer",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("Relationship Chat Analyzer")
    st.write("Upload a text file containing a chat conversation to analyze the relationship dynamics.")
    
    # Initialize classifier
    classifier = RelationshipChatClassifier()
    
    # Sidebar for options
    st.sidebar.title("Options")
    show_details = st.sidebar.checkbox("Show detailed analysis", value=True)
    
    # Try to load the model, or train a new one
    try:
        classifier.load_model()
        st.sidebar.success("Model loaded successfully!")
    except:
        st.sidebar.warning("Trained model not found. Please train the model first.")
        
        if st.sidebar.button("Train Model"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    accuracy = classifier.train("mentalmanip_con.csv")
                    classifier.save_model()
                    st.sidebar.success(f"Model trained with {accuracy:.2%} accuracy!")
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
                    st.info("Make sure the dataset file 'mentalmanip_con.csv' is in the same directory.")
    
    # File upload
    uploaded_file = st.file_uploader("Upload chat file", type="txt")
    
    col1, col2 = st.columns([3, 2])
    
    if uploaded_file is not None:
        # Read file
        chat_text = uploaded_file.getvalue().decode("utf-8")
        
        # Show chat preview in the first column
        with col1:
            st.subheader("Chat Preview")
            st.text_area("Conversation", chat_text, height=300)
        
        # Analyze button
        if st.button("Analyze Chat"):
            # Make sure model is loaded
            if classifier.model is None:
                st.error("Please train or load the model first!")
                return
                
            with st.spinner("Analyzing..."):
                # Make prediction
                result = classifier.predict(chat_text)
                
                # Display results in the second column
                with col2:
                    st.subheader("Analysis Results")
                    
                    # Show prediction with colored badge
                    label = result['label']
                    if label == "Healthy":
                        st.success(f"Prediction: {label}")
                    elif label == "Manipulative":
                        st.warning(f"Prediction: {label}")
                    else:  # "Both"
                        st.error(f"Prediction: {label}")
                    
                    st.write(f"Confidence: {result['confidence']:.2%}")
                    
                    # Show advice
                    st.subheader("Relationship Advice")
                    st.write(result['advice'])
                    
                    # Show details if requested
                    if show_details:
                        if result['techniques']:
                            st.subheader("Potential Manipulation Techniques")
                            for technique in result['techniques']:
                                st.write(f"- {technique}")
                        
                        if result['vulnerabilities']:
                            st.subheader("Potential Vulnerabilities")
                            for vulnerability in result['vulnerabilities']:
                                st.write(f"- {vulnerability}")
    
    # Example usage section
    st.sidebar.markdown("---")
    st.sidebar.subheader("How to use:")
    st.sidebar.markdown("""
    1. Upload a .txt file containing chat messages
    2. Click "Analyze Chat"
    3. View the prediction and advice
    """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("This tool analyzes relationship conversations for signs of manipulation or toxic communication patterns. It is for educational purposes only and should not replace professional help.")

if __name__ == "__main__":
    main()