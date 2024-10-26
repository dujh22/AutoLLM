import axios, { AxiosInstance } from 'axios';

// 创建 Axios 实例
const request: AxiosInstance = axios.create({
  timeout: 600000,
});

export default request;
