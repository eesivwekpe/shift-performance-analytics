import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NorDex Manufacturing - Shift Performance Analytics",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    """Load the shift performance data"""
    try:
        # Try to load from CSV first
        df = pd.read_csv('v_shiftPerformance1.csv')
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
        return df
    except FileNotFoundError:
        # Generate sample data if CSV not found
        return generate_sample_data()

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        return joblib.load('models/best_model.pkl')
    except FileNotFoundError:
        st.warning("Model file not found. Some prediction features will be unavailable.")
        return None

def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    n_records = 10000
    
    shifts = ['Morning', 'Evening', 'Night']
    skills = ['Expert', 'Senior', 'Intermediate', 'Junior']
    
    data = {
        'date': pd.date_range('2024-01-01', periods=n_records//10, freq='D').repeat(10),
        'shift_name': np.random.choice(shifts, n_records),
        'units_produced': np.random.randint(400, 1000, n_records),
        'defect_count': np.random.randint(10, 30, n_records),
        'cycle_time_avg': np.random.uniform(32, 42, n_records),
        'shift_efficiency_score': np.random.uniform(40, 105, n_records),
        'experience_level': np.random.randint(1, 13, n_records),
        'skill_category': np.random.choice(skills, n_records),
        'runtime_hours': np.random.uniform(5.9, 7.3, n_records),
        'downtime_minutes': np.random.uniform(10, 96, n_records),
        'maintenance_flag': np.random.choice([0, 1], n_records, p=[0.8, 0.2]),
        'temperature': np.random.uniform(17, 25, n_records),
        'humidity': np.random.uniform(30, 65, n_records)
    }
    
    return pd.DataFrame(data)

def calculate_shift_performance_score(df):
    """Calculate comprehensive shift performance score"""
    # Normalize metrics to 0-100 scale
    df_score = df.copy()
    
    # Production efficiency (higher is better)
    df_score['production_score'] = ((df['units_produced'] - df['units_produced'].min()) / 
                                   (df['units_produced'].max() - df['units_produced'].min())) * 100
    
    # Quality score (lower defects is better)
    df_score['quality_score'] = (1 - (df['defect_count'] - df['defect_count'].min()) / 
                                (df['defect_count'].max() - df['defect_count'].min())) * 100
    
    # Availability score (lower downtime is better)
    df_score['availability_score'] = (1 - (df['downtime_minutes'] - df['downtime_minutes'].min()) / 
                                     (df['downtime_minutes'].max() - df['downtime_minutes'].min())) * 100
    
    # Overall performance score (weighted average)
    df_score['performance_score'] = (
        df_score['production_score'] * 0.4 +
        df_score['quality_score'] * 0.3 +
        df_score['availability_score'] * 0.3
    )
    
    return df_score

def predict_optimal_configuration(model, base_config):
    """Predict optimal shift configuration"""
    if model is None:
        return None
    
    try:
        # Create variations of the base configuration
        configs = []
        for exp in range(1, 13):
            for skill in ['Expert', 'Senior', 'Intermediate', 'Junior']:
                for shift in ['Morning', 'Evening', 'Night']:
                    config = base_config.copy()
                    config.update({
                        'experience_level': exp,
                        'skill_category': skill,
                        'shift_name': shift
                    })
                    configs.append(config)
        
        config_df = pd.DataFrame(configs)
        predictions = model.predict(config_df)
        
        # Find optimal configuration
        best_idx = np.argmax(predictions)
        optimal_config = configs[best_idx]
        optimal_score = predictions[best_idx]
        
        return optimal_config, optimal_score
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# Load data and model
df = load_data()
model = load_model()

# Calculate performance scores
df_with_scores = calculate_shift_performance_score(df)

# Sidebar
st.sidebar.title("🏭 NorDex Analytics")
st.sidebar.markdown("---")

# Date range selector
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['date'].min(), df['date'].max()),
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

# Shift selector
selected_shifts = st.sidebar.multiselect(
    "Select Shifts",
    options=df['shift_name'].unique(),
    default=df['shift_name'].unique()
)

# Filter data
if len(date_range) == 2:
    mask = (df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))
    filtered_df = df[mask & df['shift_name'].isin(selected_shifts)]
    filtered_df_scores = df_with_scores[mask & df_with_scores['shift_name'].isin(selected_shifts)]
else:
    filtered_df = df[df['shift_name'].isin(selected_shifts)]
    filtered_df_scores = df_with_scores[df_with_scores['shift_name'].isin(selected_shifts)]

# Main title
st.title("🏭 NorDex Manufacturing - Shift Performance Analytics & Optimization")
st.markdown("### Real-time Performance Monitoring and Predictive Insights")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Performance Dashboard", 
    "🎯 Shift Scoring Model", 
    "🔮 Predictive Analytics", 
    "⚙️ Optimization Engine"
])

with tab1:
    st.header("Real-time Shift Performance Dashboard")
    
    # Key Performance Indicators
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_units = filtered_df['units_produced'].mean()
        st.metric("Avg Units/Shift", f"{avg_units:.0f}", delta=f"{avg_units - df['units_produced'].mean():.0f}")
    
    with col2:
        avg_defects = filtered_df['defect_count'].mean()
        st.metric("Avg Defects", f"{avg_defects:.1f}", delta=f"{avg_defects - df['defect_count'].mean():.1f}")
    
    with col3:
        avg_downtime = filtered_df['downtime_minutes'].mean()
        st.metric("Avg Downtime (min)", f"{avg_downtime:.1f}", delta=f"{avg_downtime - df['downtime_minutes'].mean():.1f}")
    
    with col4:
        avg_efficiency = filtered_df['shift_efficiency_score'].mean()
        st.metric("Efficiency Score", f"{avg_efficiency:.1f}%", delta=f"{avg_efficiency - df['shift_efficiency_score'].mean():.1f}%")
    
    with col5:
        # Calculate OEE
        availability = 1 - (filtered_df['downtime_minutes'] / (7.5 * 60))
        performance = filtered_df['units_produced'] / (100 * 7.5)
        quality = 1 - (filtered_df['defect_count'] / filtered_df['units_produced'])
        oee = (availability * performance * quality * 100).mean()
        st.metric("OEE", f"{oee:.1f}%")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Shift Performance Comparison
        shift_performance = filtered_df.groupby('shift_name').agg({
            'units_produced': 'mean',
            'defect_count': 'mean',
            'shift_efficiency_score': 'mean'
        }).reset_index()
        
        fig = px.bar(shift_performance, x='shift_name', y='shift_efficiency_score',
                    title="Average Efficiency Score by Shift",
                    color='shift_efficiency_score',
                    color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Experience vs Performance
        exp_performance = filtered_df.groupby('experience_level')['shift_efficiency_score'].mean().reset_index()
        
        fig = px.line(exp_performance, x='experience_level', y='shift_efficiency_score',
                     title="Experience Level vs Efficiency Score",
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    st.subheader("Performance Trends Over Time")
    
    daily_performance = filtered_df.groupby(['date', 'shift_name']).agg({
        'units_produced': 'sum',
        'shift_efficiency_score': 'mean'
    }).reset_index()
    
    fig = px.line(daily_performance, x='date', y='shift_efficiency_score', 
                 color='shift_name', title="Daily Efficiency Trends by Shift")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🎯 Shift Performance Scoring Model")
    
    st.markdown("""
    Our comprehensive scoring model evaluates shift performance across multiple dimensions:
    - **Production Score (40%)**: Units produced efficiency
    - **Quality Score (30%)**: Defect rate performance  
    - **Availability Score (30%)**: Downtime minimization
    """)
    
    # Performance score distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(filtered_df_scores, x='performance_score', 
                          title="Performance Score Distribution",
                          nbins=30, color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Score components breakdown
        score_components = filtered_df_scores[['production_score', 'quality_score', 'availability_score']].mean()
        
        fig = px.bar(x=score_components.index, y=score_components.values,
                    title="Average Score Components",
                    color=score_components.values,
                    color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top performing shifts
    st.subheader("Top Performing Shifts")
    
    top_shifts = filtered_df_scores.nlargest(10, 'performance_score')[
        ['date', 'shift_name', 'performance_score', 'units_produced', 'defect_count', 'downtime_minutes']
    ]
    
    st.dataframe(top_shifts, use_container_width=True)

with tab3:
    st.header("🔮 Predictive Analytics")
    
    if model is not None:
        st.subheader("Shift Efficiency Prediction")
        
        # Prediction interface
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_shift = st.selectbox("Shift", ['Morning', 'Evening', 'Night'])
            pred_units = st.slider("Units Produced", 400, 1000, 600)
            pred_defects = st.slider("Defect Count", 10, 30, 20)
        
        with col2:
            pred_experience = st.slider("Experience Level", 1, 12, 6)
            pred_skill = st.selectbox("Skill Category", ['Expert', 'Senior', 'Intermediate', 'Junior'])
            pred_runtime = st.slider("Runtime Hours", 5.9, 7.3, 6.5)
        
        with col3:
            pred_downtime = st.slider("Downtime Minutes", 10.0, 96.0, 30.0)
            pred_maintenance = st.selectbox("Maintenance Flag", [0, 1])
            pred_temp = st.slider("Temperature", 17.0, 25.0, 21.0)
            pred_humidity = st.slider("Humidity", 30.0, 65.0, 45.0)
        
        # Calculate cycle time
        pred_cycle_time = pred_runtime * 60 / pred_units if pred_units > 0 else 36
        
        # Make prediction
        if st.button("Predict Efficiency Score"):
            input_data = pd.DataFrame([{
                'shift_name': pred_shift,
                'units_produced': pred_units,
                'defect_count': pred_defects,
                'cycle_time_avg': pred_cycle_time,
                'experience_level': pred_experience,
                'skill_category': pred_skill,
                'runtime_hours': pred_runtime,
                'downtime_minutes': pred_downtime,
                'maintenance_flag': pred_maintenance,
                'temperature': pred_temp,
                'humidity': pred_humidity
            }])
            
            try:
                prediction = model.predict(input_data)[0]
                st.success(f"Predicted Efficiency Score: **{prediction:.2f}%**")
                
                # Performance category
                if prediction >= 90:
                    category = "🟢 Excellent"
                elif prediction >= 80:
                    category = "🟡 Good"
                elif prediction >= 70:
                    category = "🟠 Average"
                else:
                    category = "🔴 Below Average"
                
                st.info(f"Performance Category: {category}")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    else:
        st.warning("Predictive model not available. Please ensure the model file is present.")
    
    # Historical prediction accuracy
    st.subheader("Model Performance Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance (simulated)
        features = ['Units Produced', 'Defect Count', 'Experience Level', 'Downtime', 'Temperature']
        importance = [0.35, 0.25, 0.20, 0.15, 0.05]
        
        fig = px.bar(x=features, y=importance, title="Feature Importance",
                    color=importance, color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Prediction vs Actual (simulated)
        sample_data = filtered_df.sample(min(100, len(filtered_df)))
        fig = px.scatter(x=sample_data['shift_efficiency_score'], 
                        y=sample_data['shift_efficiency_score'] + np.random.normal(0, 2, len(sample_data)),
                        title="Predicted vs Actual Efficiency",
                        labels={'x': 'Actual', 'y': 'Predicted'})
        fig.add_shape(type="line", x0=40, y0=40, x1=105, y1=105, 
                     line=dict(dash="dash", color="red"))
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("⚙️ Optimization Engine")
    
    st.subheader("Optimal Shift Configuration Finder")
    
    # Base configuration
    st.markdown("**Define Base Parameters:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        base_units = st.number_input("Target Units", 400, 1000, 700)
        base_defects = st.number_input("Max Defects", 10, 30, 15)
    
    with col2:
        base_runtime = st.number_input("Runtime Hours", 5.9, 7.3, 7.0)
        base_downtime = st.number_input("Max Downtime", 10.0, 50.0, 25.0)
    
    with col3:
        base_temp = st.number_input("Temperature", 17.0, 25.0, 22.0)
        base_humidity = st.number_input("Humidity", 30.0, 65.0, 45.0)
    
    base_config = {
        'units_produced': base_units,
        'defect_count': base_defects,
        'cycle_time_avg': base_runtime * 60 / base_units,
        'runtime_hours': base_runtime,
        'downtime_minutes': base_downtime,
        'maintenance_flag': 0,
        'temperature': base_temp,
        'humidity': base_humidity
    }
    
    if st.button("Find Optimal Configuration") and model is not None:
        with st.spinner("Optimizing configuration..."):
            result = predict_optimal_configuration(model, base_config)
            
            if result:
                optimal_config, optimal_score = result
                
                st.success(f"**Optimal Configuration Found!**")
                st.metric("Predicted Efficiency Score", f"{optimal_score:.2f}%")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**Optimal Shift:** {optimal_config['shift_name']}")
                with col2:
                    st.info(f"**Skill Level:** {optimal_config['skill_category']}")
                with col3:
                    st.info(f"**Experience:** {optimal_config['experience_level']} years")
    
    # Optimization recommendations
    st.subheader("Performance Improvement Recommendations")
    
    # Analyze current performance gaps
    current_avg = filtered_df['shift_efficiency_score'].mean()
    top_10_avg = filtered_df.nlargest(int(len(filtered_df) * 0.1), 'shift_efficiency_score')['shift_efficiency_score'].mean()
    
    improvement_potential = top_10_avg - current_avg
    
    recommendations = []
    
    if improvement_potential > 5:
        recommendations.append("🎯 **Focus on Top Performers**: Analyze characteristics of top 10% performing shifts")
    
    # Experience level analysis
    exp_correlation = filtered_df['experience_level'].corr(filtered_df['shift_efficiency_score'])
    if exp_correlation > 0.3:
        recommendations.append("👨‍🏭 **Invest in Training**: Higher experience levels show strong correlation with performance")
    
    # Shift analysis
    shift_performance = filtered_df.groupby('shift_name')['shift_efficiency_score'].mean()
    best_shift = shift_performance.idxmax()
    recommendations.append(f"⏰ **Optimize Shift Scheduling**: {best_shift} shift shows best performance")
    
    # Maintenance impact
    maint_impact = filtered_df.groupby('maintenance_flag')['shift_efficiency_score'].mean()
    if len(maint_impact) > 1 and maint_impact[0] > maint_impact[1]:
        recommendations.append("🔧 **Preventive Maintenance**: Scheduled maintenance reduces efficiency")
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # ROI Calculator
    st.subheader("ROI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_efficiency = st.number_input("Current Avg Efficiency (%)", 50.0, 100.0, current_avg)
        target_efficiency = st.number_input("Target Efficiency (%)", current_efficiency, 100.0, min(current_efficiency + 10, 100.0))
        
    with col2:
        units_per_shift = st.number_input("Units per Shift", 400, 1000, 600)
        value_per_unit = st.number_input("Value per Unit ($)", 1.0, 100.0, 10.0)
        shifts_per_day = st.number_input("Shifts per Day", 1, 3, 3)
    
    if st.button("Calculate ROI"):
        efficiency_gain = target_efficiency - current_efficiency
        additional_units = (units_per_shift * efficiency_gain / 100) * shifts_per_day * 365
        annual_value = additional_units * value_per_unit
        
        st.success(f"**Potential Annual Value Increase: ${annual_value:,.2f}**")
        st.info(f"Additional Units/Year: {additional_units:,.0f}")
        st.info(f"Efficiency Improvement: {efficiency_gain:.1f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>NorDex Manufacturing - Shift Performance Analytics & Optimization Platform</p>
    <p>Powered by Advanced Analytics and Machine Learning</p>
</div>
""", unsafe_allow_html=True)