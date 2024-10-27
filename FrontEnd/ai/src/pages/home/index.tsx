import React, { useEffect, useState } from 'react';
import { App, Card, Empty, Input, Tabs } from 'antd';
import styles from './styles.module.scss';
import { SearchOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import SimpleData from '@/pages/home/components/simple-data';
import request from '@/utils/request.ts';
import { REQUEST_URLS } from '@/config/requet-urls.ts';
import { ISimpleData, IUpdateData } from '@/pages/home/types.ts';
import ChartData from '@/pages/home/components/chart-data';
import { ROUTE_URLS } from '@/config/route-urls.ts';
import { useAiStore } from '@/store';
import { formatBriefContent, getNewLearn, getUpdateInfo } from '@/utils/common.ts';
import { IBriefRes } from '@/types';

const HomePage: React.FC = () => {
  const nav = useNavigate();
  const { message } = App.useApp();
  const [input, setInput] = useState('');
  const [simpleDataList, setSimpleDataList] = useState<ISimpleData[]>([]);

  const phone = useAiStore(s => s.phone);

  const [updateData] = useState<IUpdateData>(getUpdateInfo());
  const [newLearn] = useState<string>(getNewLearn());

  const goToChatPage = () => {
    const url = input ? `${ROUTE_URLS.chat}?q=${encodeURIComponent(input)}` : `${ROUTE_URLS.chat}`;
    nav(url);
  };

  useEffect(() => {
    if (phone) {
      request
        .get(REQUEST_URLS.getSummary, {
          params: {
            phone_number: phone,
          },
        })
        .then(res => {
          const resData: IBriefRes = res.data;
          const briefings = Array.isArray(resData.briefings) ? resData.briefings : [resData.briefings];
          const simpleData = briefings.map(v => ({
            dateTime: new Date(v.generated_time).getTime(),
            content: formatBriefContent(v.content, v.links),
          }));
          setSimpleDataList(simpleData);
        })
        .catch(e => {
          message.error(typeof e === 'string' ? e || '网络错误' : e.toString());
        });
    }
  }, [message, phone]);

  return (
    <div className={styles.container}>
      <Card>
        <div className={styles.top}>
          <div className={styles.inputWrapper}>
            <div className={styles.advantage}>由大模型整体当日知识，提升学习效率</div>
            <Input
              className={styles.input}
              size="large"
              addonAfter={
                <div className={styles.search} onClick={() => goToChatPage()}>
                  <SearchOutlined />
                </div>
              }
              placeholder={import.meta.env.VITE_APP_SLOGAN}
              allowClear
              onChange={e => setInput(e.target.value)}
              onPressEnter={() => goToChatPage()}
            />
            <div className={styles.new}>今日新学知识点：{newLearn}</div>
          </div>
          <div className={styles.data}>
            <div className={styles.dataItem}>
              今日新知识：<span className={styles.spText}> +{updateData.newNum}</span> 篇
            </div>
            <div className={styles.dataItem}>知识汇总：{updateData.allNum} 篇</div>
          </div>
        </div>
      </Card>
      <div className={styles.content}>
        <Tabs
          className={styles.left}
          type="card"
          items={[
            {
              label: `知识简报`,
              key: 'simple',
              children: <SimpleData dataList={simpleDataList} />,
            },
            {
              label: `知识原文`,
              key: 'full',
              children: <Empty description="To be done..." />,
            },
          ]}
        />
        <div className={styles.right}>
          <ChartData />
        </div>
      </div>
    </div>
  );
};

export default HomePage;
